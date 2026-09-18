#!/usr/bin/env node
/**
 * Long-running SMPP traffic generator for soak-testing the OTS proxy.
 *
 * Sends a labeled corpus through the proxy at a diurnal rate with hourly
 * bursts, records the outcome of every message (forwarded / rejected / error,
 * latency) and every DLR relayed back, and reconnects if the proxy drops.
 *
 *   node evals/soak/soak_client.js --corpus ~/ots-soak/corpus.jsonl --out ~/ots-soak/run-X \
 *        [--host 127.0.0.1] [--port 2775] [--user client1] [--pass secret123] \
 *        [--base-rate 3] [--burst-rate 12] [--hours 48]
 *
 * Rate model: base_rate scaled by a day curve (0.25x at 04:00, 1.5x at 14:00),
 * plus a 45 second burst at burst_rate every hour at :30.
 */

const fs = require('fs')
const path = require('path')
const smpp = require('smpp')

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
	if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]])
	return acc
}, []))
const HOST = args.host || '127.0.0.1'
const PORT = Number(args.port || 2775)
const USER = args.user || 'client1'
const PASS = args.pass || 'secret123'
const BASE_RATE = Number(args['base-rate'] || 3)
const BURST_RATE = Number(args['burst-rate'] || 12)
const HOURS = Number(args.hours || 48)
const OUT = path.resolve(args.out.replace(/^~/, process.env.HOME))
fs.mkdirSync(OUT, {recursive: true})

const corpus = fs.readFileSync(path.resolve(args.corpus.replace(/^~/, process.env.HOME)), 'utf8')
	.split('\n').filter(Boolean).map(JSON.parse)
const results = fs.createWriteStream(path.join(OUT, 'results.jsonl'), {flags: 'a'})
const events = fs.createWriteStream(path.join(OUT, 'client_events.log'), {flags: 'a'})
const summaryPath = path.join(OUT, 'summary.log')

const stats = {sent: 0, forwarded: 0, rejected: 0, error: 0, timeout: 0, dlr: 0, reconnects: 0,
	byLabel: {}, latencies: []}
const startedAt = Date.now()
const endAt = startedAt + HOURS * 3600 * 1000
let session = null
let bound = false
let inFlight = 0
let cursor = 0
let stopping = false

function log(msg, extra) {
	const line = `${new Date().toISOString()} ${msg}${extra ? ' ' + JSON.stringify(extra) : ''}`
	events.write(line + '\n')
	console.log(line)
}

function rateNow() {
	const h = new Date().getHours() + new Date().getMinutes() / 60
	// day curve: trough 0.25x at 04:00, peak 1.5x at 14:00
	const curve = 0.875 + 0.625 * Math.cos((h - 14) / 24 * 2 * Math.PI)
	const m = new Date().getMinutes(), s = new Date().getSeconds()
	const burst = (m === 30 && s < 45) ? BURST_RATE : 0
	return BASE_RATE * curve + burst
}

function connect() {
	if (stopping) return
	session = smpp.connect({url: `smpp://${HOST}:${PORT}`}, () => {
		session.bind_transceiver({system_id: USER, password: PASS}, (pdu) => {
			if (pdu.command_status !== 0) {
				log('bind failed', {status: pdu.command_status})
				session.close()
				return
			}
			bound = true
			log('bound', {host: HOST, port: PORT})
		})
	})
	session.on('enquire_link', (pdu) => session.send(pdu.response()))
	session.on('deliver_sm', (pdu) => {
		stats.dlr++
		session.send(pdu.response())
	})
	session.on('error', (e) => log('session error', {error: e.message}))
	session.on('close', () => {
		bound = false
		if (stopping) return
		stats.reconnects++
		log('session closed, reconnecting in 5s')
		setTimeout(connect, 5000)
	})
	// keep the bind alive across quiet minutes at night
	setInterval(() => { if (bound) session.enquire_link() }, 30000).unref()
}

function submitOne() {
	if (!bound || cursor >= corpus.length) return
	const item = corpus[cursor++]
	if (cursor >= corpus.length) cursor = 0  // loop the corpus for multi-day runs
	const pdu = {source_addr: item.pool === 'ham_personal' ? '4479' + String(item.id % 900000 + 100000) : 'OTS' + (item.id % 97),
		destination_addr: '4477' + String(100000 + (item.id * 7919) % 900000)}
	// short_message carries at most 255 bytes: non-Latin text is UCS-2 (2 bytes per
	// character), so anything a real SMSC would split goes via message_payload.
	const nonAscii = /[^\x00-\x7F]/.test(item.text)
	const usePayload = item.via === 'payload' || item.text.length > 160 || (nonAscii && item.text.length > 70)
	if (usePayload) pdu.message_payload = item.text
	else pdu.short_message = item.text
	item.via = usePayload ? 'payload' : 'short'
	const t0 = Date.now()
	inFlight++
	stats.sent++
	let done = false
	const timer = setTimeout(() => {
		if (done) return
		done = true
		inFlight--
		stats.timeout++
		record(item, 'timeout', null, Date.now() - t0)
	}, 30000)
	try {
		session.submit_sm(pdu, (resp) => {
			if (done) return
			done = true
			clearTimeout(timer)
			inFlight--
			const ms = Date.now() - t0
			const outcome = resp.command_status === 0 ? 'forwarded' : resp.command_status === 69 ? 'rejected' : 'error'
			stats[outcome]++
			record(item, outcome, resp.command_status, ms)
		})
	} catch (e) {
		done = true
		clearTimeout(timer)
		inFlight--
		stats.error++
		record(item, 'error', -1, Date.now() - t0, e.message)
	}
}

function record(item, outcome, status, ms, err) {
	const key = item.label
	stats.byLabel[key] = stats.byLabel[key] || {sent: 0, forwarded: 0, rejected: 0, error: 0, timeout: 0}
	stats.byLabel[key].sent++
	stats.byLabel[key][outcome]++
	stats.latencies.push(ms)
	if (stats.latencies.length > 5000) stats.latencies.shift()
	results.write(JSON.stringify({ts: new Date().toISOString(), id: item.id, label: item.label, pool: item.pool,
		source: item.source, lang: item.lang, category: item.category, obf: item.obfuscation, via: item.via,
		len: item.text.length, outcome, status, ms, err}) + '\n')
}

function pct(arr, p) {
	if (!arr.length) return null
	const s = [...arr].sort((a, b) => a - b)
	return s[Math.min(s.length - 1, Math.floor(p * s.length))]
}

function writeSummary() {
	const line = {ts: new Date().toISOString(), uptime_h: +((Date.now() - startedAt) / 3600000).toFixed(2),
		rate_now: +rateNow().toFixed(2), in_flight: inFlight, bound, ...stats,
		latency_p50: pct(stats.latencies, 0.5), latency_p95: pct(stats.latencies, 0.95), latency_p99: pct(stats.latencies, 0.99)}
	delete line.latencies
	fs.appendFileSync(summaryPath, JSON.stringify(line) + '\n')
}

// scheduler: Poisson-ish arrivals at rateNow(), bounded in-flight
function tick() {
	if (stopping) return
	if (Date.now() >= endAt) { stop('duration reached'); return }
	const r = rateNow()
	if (bound && inFlight < 16 && r > 0) submitOne()
	const wait = -Math.log(1 - Math.random()) / r * 1000
	setTimeout(tick, Math.max(5, Math.min(wait, 5000)))
}

function stop(reason) {
	stopping = true
	log('stopping', {reason})
	writeSummary()
	setTimeout(() => { try { session && session.close() } catch (e) {} ; process.exit(0) }, 1500)
}

process.on('SIGTERM', () => stop('SIGTERM'))
process.on('SIGINT', () => stop('SIGINT'))

log('soak client starting', {corpus: corpus.length, base_rate: BASE_RATE, burst_rate: BURST_RATE, hours: HOURS, out: OUT})
connect()
setInterval(writeSummary, 5 * 60 * 1000)
tick()
