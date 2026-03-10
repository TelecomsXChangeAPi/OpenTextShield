/**
 * OpenTextShield SMPP Interface
 *
 * SMPP proxy that classifies SMS messages using the OTS mBERT model
 * before forwarding to an upstream SMSC.
 *
 * Architecture:
 *   [Client SMSC] --> [OTS SMPP Server] --> [OTS API classify] --> [Upstream SMSC]
 *
 * Based on TCXC SMPP proxy patterns (proxy_async.js + server-balancer.go)
 */

const smpp = require('smpp')
const sleep = require('await-sleep')
const {customAlphabet} = require('nanoid')
const http = require('http')
const fs = require('fs')
const path = require('path')

// Load config
const configPath = process.argv[2] || path.join(__dirname, 'config.json')
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))

// Parse classification API URL(s) — supports single URL or array for load balancing
let apiUrls = []
if (Array.isArray(config.classification.api_urls)) {
	apiUrls = config.classification.api_urls.map(u => new URL(u))
} else if (config.classification.api_url) {
	apiUrls = [new URL(config.classification.api_url)]
}
let apiUrlIndex = 0

// ─────────────────────────────────────────────
// Logging (ref: proxy_async.js:306-312)
// ─────────────────────────────────────────────

const logsDir = path.dirname(path.resolve(__dirname, config.logging.file))
if (!fs.existsSync(logsDir)) {
	fs.mkdirSync(logsDir, { recursive: true })
}

let logFile = fs.createWriteStream(path.resolve(__dirname, config.logging.file), { flags: 'a' })

function llog(func, params, msg) {
	let today = new Date().toJSON().slice(0, 23).replace('T', ' ')
	let log_msg = today + ' [' + func + '] ' + JSON.stringify(msg) + ' ' + JSON.stringify(params)
	logFile.write(log_msg + '\n')
	if (config.logging.level === 'debug') {
		console.log(log_msg)
	}
}

// Redirect uncaught console output to log file (after startup messages)
// This captures node-smpp internal logging and unexpected console.log calls
let _origStdoutWrite = process.stdout.write.bind(process.stdout)
let _origStderrWrite = process.stderr.write.bind(process.stderr)

function redirectOutputToLog() {
	process.stdout.write = function(chunk) {
		logFile.write(chunk)
	}
	process.stderr.write = function(chunk) {
		logFile.write(chunk)
	}
}

// PID file
let pidFile = path.join(__dirname, 'ots_smpp_proxy.pid')
fs.writeFileSync(pidFile, '' + process.pid)

// ─────────────────────────────────────────────
// Process handlers (ref: proxy_async.js:314-324)
// ─────────────────────────────────────────────

process.on('uncaughtException', function(err) {
	console.error('Uncaught Exception:', (err && err.stack) ? err.stack : err)
})

process.on('SIGHUP', () => {
	llog('signal', {}, 'GOT HUP SIGNAL - rotating log')
	logFile.close()
	logFile = fs.createWriteStream(path.resolve(__dirname, config.logging.file), { flags: 'a' })
	redirectOutputToLog()
})

function cleanShutdown(signal) {
	llog('shutdown', {signal: signal}, 'Received ' + signal + ', shutting down')
	// Clean up PID file
	try { fs.unlinkSync(pidFile) } catch(e) {}
	// Close upstream sessions gracefully
	for (let s of upstream_pool) {
		try { s.close() } catch(e) {}
	}
	process.exit(0)
}

process.on('SIGTERM', () => cleanShutdown('SIGTERM'))
process.on('SIGINT', () => cleanShutdown('SIGINT'))

// ─────────────────────────────────────────────
// Nanoid helper
// ─────────────────────────────────────────────

function nanoid(len = 8) {
	const nanoid1 = customAlphabet('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', len)
	return nanoid1()
}

// ─────────────────────────────────────────────
// Session tracking (ref: proxy_async.js:127-131)
// ─────────────────────────────────────────────

var isessions = {}                    // client sessions: nickname -> session
var upstream_pool = []                // upstream session pool (array)
var upstream_sessions_timeouts = {}   // upstream session timeout tracking

// ─────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────

var stats = {
	messages_received: 0,
	messages_classified: 0,
	messages_forwarded: 0,
	messages_rejected: 0,
	messages_dropped: 0,
	dlrs_forwarded: 0,
	classification_errors: 0,
	active_sessions: 0,
	upstream_pool_size: 0
}

// Log stats periodically
setInterval(function() {
	stats.upstream_pool_size = upstream_pool.length
	stats.message_store_size = Object.keys(message_store).length
	llog('stats', stats, 'Periodic stats')
}, 60000)

// ─────────────────────────────────────────────
// In-memory message store for DLR correlation
// (replaces MySQL smpp_messages table)
// ─────────────────────────────────────────────

var message_store = {}

// Periodic cleanup of old entries >24h (ref: proxy_async.js DLR lookup with "ts > NOW() - interval 1 day")
// Runs every 10 minutes instead of hourly to limit memory growth
const MESSAGE_STORE_MAX_SIZE = 500000 // safety cap: ~60MB
const MESSAGE_STORE_TTL = 86400000    // 24 hours

setInterval(function() {
	let now = Date.now()
	let cleaned = 0
	let store_size = Object.keys(message_store).length

	// Normal TTL cleanup
	for (let key in message_store) {
		if (now - message_store[key].timestamp > MESSAGE_STORE_TTL) {
			delete message_store[key]
			cleaned++
		}
	}

	// Emergency size cap — remove oldest entries if over limit
	if (store_size - cleaned > MESSAGE_STORE_MAX_SIZE) {
		let entries = Object.entries(message_store).sort((a, b) => a[1].timestamp - b[1].timestamp)
		let excess = entries.length - MESSAGE_STORE_MAX_SIZE
		for (let i = 0; i < excess; i++) {
			delete message_store[entries[i][0]]
			cleaned++
		}
		llog('cleanup', {excess: excess}, 'Emergency cleanup - message store exceeded max size')
	}

	if (cleaned > 0) {
		llog('cleanup', {cleaned: cleaned, remaining: Object.keys(message_store).length}, 'Message store cleanup')
	}
}, 600000) // every 10 minutes

// ─────────────────────────────────────────────
// Promise wrappers (ref: proxy_async.js:234-294)
// ─────────────────────────────────────────────

let promise_bind = function(session, params, connect_timeout) {
	return new Promise((resolve, reject) => {
		const timeoutObj = setTimeout(() => {
			let pdu = {command_status: -1}
			resolve(pdu)
		}, connect_timeout * 1000)

		session.on('error', function(pdu) {
			llog('promise_bind', {pdu: pdu}, 'Session error during bind')
			pdu = pdu || {}
			pdu.command_status = -1
			clearTimeout(timeoutObj)
			resolve(pdu)
		})

		session.on('close', function() {
			llog('promise_bind', {}, 'Session closed during bind')
			let pdu = {command_status: -1}
			clearTimeout(timeoutObj)
			resolve(pdu)
		})

		session.bind_transceiver(params, (pdu) => {
			clearTimeout(timeoutObj)
			resolve(pdu)
		})
	})
}

let promise_submit_sm = function(session, params, submit_sm_timeout) {
	return new Promise((resolve, reject) => {
		// 0xFE = Transaction delivery failure (ref: proxy_async.js:264-266)
		const timeoutObj = setTimeout(() => {
			let pdu = {command_status: 254}
			resolve(pdu)
		}, submit_sm_timeout * 1000)

		try {
			session.submit_sm(params, (pdu) => {
				clearTimeout(timeoutObj)
				resolve(pdu)
			})
		} catch(e) {
			clearTimeout(timeoutObj)
			console.log('Exception during submit_sm:', e)
			let rpdu = {command_status: 18} // system error
			resolve(rpdu)
		}
	})
}

let promise_deliver_sm = function(session, params) {
	return new Promise((resolve, reject) => {
		try {
			session.deliver_sm(params, (pdu) => {
				resolve(pdu)
			})
		} catch(e) {
			console.log('Exception during deliver_sm:', e)
			let rpdu = {command_status: 18}
			resolve(rpdu)
		}
	})
}

// ─────────────────────────────────────────────
// OTS Classification via HTTP API
// (replaces authorizeAccount from reference)
// ─────────────────────────────────────────────

async function classifyMessage(text) {
	// Round-robin across API instances for load balancing
	const url = apiUrls[apiUrlIndex % apiUrls.length]
	apiUrlIndex = (apiUrlIndex + 1) % apiUrls.length

	return new Promise((resolve, reject) => {
		const data = JSON.stringify({
			text: text,
			model: config.classification.model
		})

		const options = {
			hostname: url.hostname,
			port: url.port,
			path: url.pathname,
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Content-Length': Buffer.byteLength(data)
			},
			timeout: config.classification.timeout
		}

		const req = http.request(options, (res) => {
			let body = ''
			res.on('data', chunk => body += chunk)
			res.on('end', () => {
				if (res.statusCode !== 200) {
					reject(new Error('Classification API returned HTTP ' + res.statusCode + ': ' + body.substring(0, 200)))
					return
				}
				try {
					const result = JSON.parse(body)
					resolve({
						label: result.label,
						probability: result.probability,
						processing_time: result.processing_time
					})
				} catch(e) {
					reject(new Error('Failed to parse classification response: ' + e.message))
				}
			})
		})

		req.on('error', (e) => {
			reject(new Error('Classification API error (' + url.host + '): ' + e.message))
		})

		req.on('timeout', () => {
			req.destroy()
			reject(new Error('Classification API timeout (' + url.host + ')'))
		})

		req.write(data)
		req.end()
	})
}

// ─────────────────────────────────────────────
// Message text extraction + UDH handling
// (ref: proxy_async.js:759-796 get_log_text + 1599-1618 UDH)
// ─────────────────────────────────────────────

function getMessageText(pdu) {
	let text = ''

	// Try short_message first (ref: proxy_async.js:769-773)
	if (pdu.short_message != undefined) {
		if (pdu.short_message.message != undefined && pdu.short_message.message != '') {
			text = pdu.short_message.message
		}
	}

	// Fallback to message_payload TLV (ref: proxy_async.js:764-768)
	if (text == '' && ('message_payload' in pdu)) {
		if (pdu.message_payload != undefined) {
			if (pdu.message_payload.message != undefined && pdu.message_payload.message != '') {
				text = pdu.message_payload.message
			} else if (typeof pdu.message_payload === 'string' || Buffer.isBuffer(pdu.message_payload)) {
				text = pdu.message_payload
			}
		}
	}

	// UDH scenario — text is in .message part, UDH is separate
	// (ref: proxy_async.js:775-782)
	if (pdu.short_message && pdu.short_message.udh != undefined) {
		if (Buffer.isBuffer(pdu.short_message.message)) {
			text = pdu.short_message.message.toString()
		} else {
			text = pdu.short_message.message
		}
	}

	// Handle Buffer (UCS-2 / binary data)
	if (Buffer.isBuffer(text)) {
		if (pdu.data_coding == 8) {
			text = text.toString('ucs2')
		} else {
			text = text.toString()
		}
	}

	return text || ''
}

// Build upstream PDU preserving UDH (ref: proxy_async.js:1599-1618)
function buildUpstreamPdu(pdu) {
	let upstream_pdu = {
		source_addr: pdu.source_addr,
		destination_addr: pdu.destination_addr,
		data_coding: pdu.data_coding,
		esm_class: pdu.esm_class,
		registered_delivery: pdu.registered_delivery
	}

	// Preserve TON/NPI if set
	if ('source_addr_ton' in pdu) upstream_pdu.source_addr_ton = pdu.source_addr_ton
	if ('source_addr_npi' in pdu) upstream_pdu.source_addr_npi = pdu.source_addr_npi
	if ('dest_addr_ton' in pdu) upstream_pdu.dest_addr_ton = pdu.dest_addr_ton
	if ('dest_addr_npi' in pdu) upstream_pdu.dest_addr_npi = pdu.dest_addr_npi

	// UDH handling (ref: proxy_async.js:1599-1618)
	if (pdu.short_message && pdu.short_message.udh != undefined) {
		let temp_udh = pdu.short_message.udh
		if (Array.isArray(temp_udh)) {
			temp_udh = Buffer.concat(temp_udh)
			let len_buf = Buffer.alloc(1)
			len_buf.writeUInt8(temp_udh.length, 0)
			temp_udh = Buffer.concat([len_buf, temp_udh])
		}
		upstream_pdu.short_message = {udh: temp_udh, message: pdu.short_message.message}
	} else if (pdu.short_message) {
		upstream_pdu.short_message = pdu.short_message
	}

	// message_payload passthrough
	if (pdu.message_payload) {
		upstream_pdu.message_payload = pdu.message_payload
	}

	return upstream_pdu
}

// ─────────────────────────────────────────────
// DLR Handling
// (ref: proxy_async.js:383-481 deliver_sm handler)
// ─────────────────────────────────────────────

function handleDLR(pdu) {
	let report_string = ''

	if (pdu.short_message) {
		report_string = pdu.short_message.message || ''
	}

	// Fallback to message_payload (ref: proxy_async.js:392-396)
	if (report_string == '' && ('message_payload' in pdu)) {
		if (pdu.message_payload && pdu.message_payload.message) {
			report_string = pdu.message_payload.message
		}
	}

	if (Buffer.isBuffer(report_string)) {
		report_string = report_string.toString()
	}

	// Extract message_id from DLR text (ref: proxy_async.js:397-401)
	let re = /id:([^ ]+) /
	let found_id = (typeof report_string === 'string') ? report_string.match(re) : null

	if (!found_id || found_id.length < 2) {
		llog('dlr', {report: report_string}, 'Unmatched DLR - no message_id')
		return
	}

	let message_id = found_id[1]
	let mapping = message_store[message_id]

	// Try hex conversion if not found (ref: proxy_async.js:439-443)
	if (!mapping && /^\d+$/.test(message_id)) {
		try {
			let hex_id = BigInt(message_id).toString(16)
			mapping = message_store[hex_id] || message_store['0' + hex_id]
			if (mapping) {
				llog('dlr', {original: message_id, hex: hex_id}, 'Matched via hex conversion')
			}
		} catch(e) {
			// BigInt conversion failed, skip
		}
	}

	if (!mapping) {
		llog('dlr', {message_id: message_id}, 'DLR for unknown message_id')
		return
	}

	let client_session = isessions[mapping.session_id]
	if (client_session) {
		try {
			client_session.deliver_sm(pdu)
			stats.dlrs_forwarded++
			llog('dlr', {message_id: message_id, client: mapping.session_id}, 'DLR forwarded to client')
		} catch(err) {
			llog('dlr', {message_id: message_id, error: err.message}, 'Failed to forward DLR')
		}
	} else {
		llog('dlr', {message_id: message_id, session_id: mapping.session_id}, 'Client session not found for DLR')
	}

	// Clean up
	delete message_store[message_id]
}

// ─────────────────────────────────────────────
// Upstream Session Pool
// (ref: Go server-balancer.go:196-298 makePool/makeSession)
// (ref: Node.js proxy_async.js:488-670 get_session)
// ─────────────────────────────────────────────

async function createUpstreamPool() {
	let num_connections = config.upstream.connections || 1
	llog('upstream', {connections: num_connections, host: config.upstream.host, port: config.upstream.port},
		'Creating upstream session pool')

	for (let i = 0; i < num_connections; i++) {
		let session = await createUpstreamSession(i)
		if (session) {
			upstream_pool.push(session)
		}
	}
	llog('upstream', {pool_size: upstream_pool.length}, 'Upstream pool ready')
}

async function createUpstreamSession(index) {
	// Interface version (ref: proxy_async.js:496-499)
	let iface_version = 0x50
	if (config.upstream.interface_version == '3.4') {
		iface_version = 0x34
	}

	try {
		let session = await smpp.connect({
			url: 'smpp://' + config.upstream.host + ':' + config.upstream.port
		})

		let bind_pdu = await promise_bind(session, {
			system_id: config.upstream.system_id,
			password: config.upstream.password,
			interface_version: iface_version
		}, config.upstream.connect_timeout)

		if (bind_pdu.command_status != 0) {
			throw new Error('Bind failed with status: ' + bind_pdu.command_status)
		}

		let nick = 'UP-' + index + '-' + nanoid(8)
		session.nickname = nick
		session.pool_index = index

		// Enquire_link handler (ref: proxy_async.js:533-535)
		session.on('enquire_link', function(pdu) {
			session.send(pdu.response())
		})

		// DLR handler (ref: proxy_async.js:586-668)
		session.on('deliver_sm', async function(pdu) {
			session.send(pdu.response())
			if (pdu.esm_class == 4) {
				handleDLR(pdu)
			}
		})

		// Close handler — auto-reconnect (ref: Go server-balancer.go:228-239)
		session.on('close', function() {
			llog('upstream', {nickname: nick}, 'Session closed, reconnecting...')
			removeFromPool(nick)
			clearTimeout(session.ka_timeout)
			clearTimeout(session.s_timeout)
			delete upstream_sessions_timeouts[nick]
			setTimeout(async function() {
				await reconnectUpstreamSession(index)
			}, config.upstream.reconnect_delay * 1000)
		})

		session.on('error', function(err) {
			llog('upstream', {nickname: nick, error: (err && err.message) ? err.message : err}, 'Session error')
			removeFromPool(nick)
		})

		session.on('unbind', function(pdu) {
			llog('upstream', {nickname: nick}, 'Received unbind from upstream')
			session.close()
		})

		// Keepalive setup (ref: proxy_async.js:580-584)
		const now = Math.floor(new Date() / 1000)
		upstream_sessions_timeouts[nick] = now + config.upstream.session_timeout

		if (config.upstream.enquire_link_interval > 0) {
			session.ka_timeout = setTimeout(upstreamKeepAlive,
				config.upstream.enquire_link_interval * 1000, nick, session)
		}
		session.s_timeout = setTimeout(upstreamSessionTimeout,
			config.upstream.session_timeout * 1000 + 1000, nick, session)

		llog('upstream', {nickname: nick, index: index}, 'Connected and bound')
		return session

	} catch(err) {
		llog('upstream', {error: err.message, index: index}, 'Connect failed')
		return null
	}
}

// Reconnect with retry and backoff (ref: Go server-balancer.go:232-236)
const MAX_RECONNECT_ATTEMPTS = 50
const MAX_RECONNECT_DELAY = 60 // cap at 60 seconds

async function reconnectUpstreamSession(index) {
	let session = null
	let attempts = 0
	while (session == null && attempts < MAX_RECONNECT_ATTEMPTS) {
		session = await createUpstreamSession(index)
		if (session) {
			upstream_pool.push(session)
			llog('upstream', {index: index, pool_size: upstream_pool.length}, 'Reconnected')
		} else {
			attempts++
			// Exponential backoff: delay * 2^(attempt-1), capped at MAX_RECONNECT_DELAY
			let delay = Math.min(config.upstream.reconnect_delay * Math.pow(2, Math.min(attempts - 1, 5)), MAX_RECONNECT_DELAY)
			llog('upstream', {index: index, attempt: attempts, delay: delay},
				'Reconnect failed, retrying in ' + delay + 's')
			await sleep(delay * 1000)
		}
	}
	if (session == null) {
		llog('upstream', {index: index, attempts: attempts}, 'GAVE UP reconnecting after max attempts')
	}
}

// Random selection from pool (ref: Go getRandomSession at server-balancer.go:77-82)
function getRandomUpstreamSession() {
	if (upstream_pool.length == 0) return null
	let index = Math.floor(Math.random() * upstream_pool.length)
	return upstream_pool[index]
}

function removeFromPool(nickname) {
	upstream_pool = upstream_pool.filter(s => s.nickname != nickname)
}

// Keepalive for upstream sessions (ref: proxy_async.js:688-705)
function upstreamKeepAlive(ka_key, session) {
	const now = Math.floor(new Date() / 1000)
	if (session.socket && !session.socket.destroyed) {
		session.enquire_link(function(pdu) {
			if (pdu.command_status == 0) {
				upstream_sessions_timeouts[ka_key] = now + config.upstream.session_timeout
				llog('upstream_ka', {nickname: ka_key}, 'enquire_link OK')
			} else {
				llog('upstream_ka', {nickname: ka_key}, 'enquire_link FAILED')
			}
		})
		session.ka_timeout = setTimeout(upstreamKeepAlive,
			config.upstream.enquire_link_interval * 1000, ka_key, session)
	}
}

// Timeout check for upstream (ref: proxy_async.js:672-686)
function upstreamSessionTimeout(ka_key, session) {
	const now = Math.floor(new Date() / 1000)
	if (session.socket && !session.socket.destroyed
		&& upstream_sessions_timeouts[ka_key] != undefined
		&& upstream_sessions_timeouts[ka_key] < now) {
		llog('upstream_timeout', {nickname: ka_key}, 'Session timed out, closing')
		session.close()
		return
	}
	if (session.socket && !session.socket.destroyed) {
		session.s_timeout = setTimeout(upstreamSessionTimeout,
			config.upstream.session_timeout * 1000 + 1000, ka_key, session)
	}
}

// ─────────────────────────────────────────────
// Forward to upstream
// (replaces uponAuthSubmit from reference)
// ─────────────────────────────────────────────

async function forwardToUpstream(isession, pdu) {
	let session = getRandomUpstreamSession()
	if (!session) {
		isession.send(pdu.response({command_status: 0x08})) // ESME_RSYSERR
		llog('forward', {}, 'No upstream sessions available')
		return
	}

	// Reset upstream timeout on activity
	upstream_sessions_timeouts[session.nickname] = Math.floor(new Date() / 1000) + config.upstream.session_timeout

	// Build upstream PDU with proper UDH handling
	let upstream_pdu = buildUpstreamPdu(pdu)

	try {
		let resp = await promise_submit_sm(session, upstream_pdu, config.upstream.submit_sm_timeout)

		if (resp.command_status == 0) {
			let message_id = resp.message_id
			// Store for DLR correlation
			message_store[message_id] = {
				session_id: isession.nickname,
				source_addr: pdu.source_addr,
				destination_addr: pdu.destination_addr,
				timestamp: Date.now()
			}
			// Return upstream's message_id to client
			isession.send(pdu.response({message_id: message_id}))
			llog('forward', {message_id: message_id, dst: pdu.destination_addr, via: session.nickname}, 'Forwarded OK')
		} else {
			// Forward the error status
			isession.send(pdu.response({command_status: resp.command_status}))
			llog('forward', {status: resp.command_status, dst: pdu.destination_addr}, 'Upstream rejected')
		}
	} catch(err) {
		isession.send(pdu.response({command_status: 0x08})) // ESME_RSYSERR
		llog('forward', {error: err.message}, 'Forward failed')
	}
}

// ─────────────────────────────────────────────
// SMPP Server - accepts client binds
// (ref: proxy_async.js:1865-1957)
// ─────────────────────────────────────────────

let server = smpp.createServer(function(isession) {

	// ── Bind handler (ref: proxy_async.js:1866-1900) ──
	isession.on('bind_transceiver', async function(pdu) {
		let log_pdu = Object.assign({}, pdu)
		if (log_pdu.password != undefined) {
			log_pdu.password = 'HIDDEN'
		}

		// Pause session during auth (ref: proxy_async.js:1875)
		isession.pause()

		let system_id = pdu.system_id
		let password = pdu.password

		// Check credentials against config
		let client = config.server.clients[system_id]
		if (!client || client.password != password) {
			isession.send(pdu.response({
				command_status: smpp.ESME_RBINDFAIL
			}))
			llog('bind', {system_id: system_id, ip: isession.ip && isession.ip()}, 'REJECTED - invalid credentials')
			isession.close()
			return
		}

		// Accept bind
		isession.send(pdu.response({system_id: config.server.system_id}))
		const nickname = nanoid(24)
		isession.nickname = nickname
		isession.system_id_str = system_id
		isessions[nickname] = isession
		isession.resume()

		stats.active_sessions++
		llog('bind', {system_id: system_id, nickname: nickname, ip: isession.ip && isession.ip()}, 'AUTHENTICATED')

		// Client keepalive — send enquire_link to detect dead connections
		isession._last_activity = Date.now()
		if (config.server.enquire_link_interval > 0) {
			isession._ka_interval = setInterval(function() {
				if (!isession.socket || isession.socket.destroyed) {
					clearInterval(isession._ka_interval)
					return
				}
				isession.enquire_link(function(pdu) {
					if (pdu && pdu.command_status == 0) {
						isession._last_activity = Date.now()
					}
				})
			}, config.server.enquire_link_interval * 1000)
		}

		// Client session timeout — close if no activity
		if (config.server.session_timeout > 0) {
			isession._timeout_interval = setInterval(function() {
				if (!isession.socket || isession.socket.destroyed) {
					clearInterval(isession._timeout_interval)
					return
				}
				let idle_sec = (Date.now() - isession._last_activity) / 1000
				if (idle_sec > config.server.session_timeout) {
					llog('client_timeout', {nickname: nickname, idle: idle_sec}, 'Client session timed out')
					isession.close()
				}
			}, config.server.session_timeout * 1000)
		}
	})

	// ── Reject bind_receiver / bind_transmitter — we only support transceiver ──
	isession.on('bind_receiver', function(pdu) {
		llog('bind', {system_id: pdu.system_id}, 'REJECTED bind_receiver - use bind_transceiver')
		isession.send(pdu.response({command_status: smpp.ESME_RBINDFAIL}))
		isession.close()
	})

	isession.on('bind_transmitter', function(pdu) {
		llog('bind', {system_id: pdu.system_id}, 'REJECTED bind_transmitter - use bind_transceiver')
		isession.send(pdu.response({command_status: smpp.ESME_RBINDFAIL}))
		isession.close()
	})

	// ── Enquire_link handler (ref: proxy_async.js:1913-1916) ──
	isession.on('enquire_link', function(pdu) {
		isession._last_activity = Date.now()
		isession.send(pdu.response())
	})

	// ── Unbind handler (ref: proxy_async.js:1926-1929) ──
	isession.on('unbind', function(pdu) {
		llog('unbind', {nickname: isession.nickname}, 'Client unbind')
		isession.send(pdu.response())
		isession.close()
	})

	// ── Close handler (ref: proxy_async.js:1903-1911) ──
	isession.on('close', function() {
		if (isession._ka_interval) clearInterval(isession._ka_interval)
		if (isession._timeout_interval) clearInterval(isession._timeout_interval)
		if (isession.nickname) {
			llog('close', {nickname: isession.nickname}, 'Client session closed')
			delete isessions[isession.nickname]
			stats.active_sessions = Math.max(0, stats.active_sessions - 1)
		}
	})

	// ── Error handler (ref: proxy_async.js:1918-1924) ──
	isession.on('error', function(err) {
		if (isession._ka_interval) clearInterval(isession._ka_interval)
		if (isession._timeout_interval) clearInterval(isession._timeout_interval)
		llog('session_error', {nickname: isession.nickname, error: (err && err.message) ? err.message : err}, 'Session error')
		if (isession.nickname) {
			delete isessions[isession.nickname]
			stats.active_sessions = Math.max(0, stats.active_sessions - 1)
		}
	})

	// ── Submit_sm handler - CORE CLASSIFICATION FLOW ──
	// (replaces authorizeAccount + uponAuthSubmit from reference)
	isession.on('submit_sm', async function(pdu) {
		llog('submit_sm', {
			src: pdu.source_addr,
			dst: pdu.destination_addr,
			nickname: isession.nickname,
			data_coding: pdu.data_coding,
			esm_class: pdu.esm_class
		}, 'Received')

		isession._last_activity = Date.now()
		stats.messages_received++

		// 1. Extract message text (handles UDH, message_payload, data_coding)
		let text = getMessageText(pdu)
		if (text == '') {
			llog('submit_sm', {dst: pdu.destination_addr}, 'Empty message, forwarding as ham')
			await forwardToUpstream(isession, pdu)
			stats.messages_forwarded++
			return
		}

		// 2. Classify with OTS API
		let classification
		try {
			classification = await classifyMessage(text)
			stats.messages_classified++
		} catch(err) {
			llog('submit_sm', {error: err.message}, 'Classification failed, forwarding as ham (fail-open)')
			stats.classification_errors++
			await forwardToUpstream(isession, pdu)
			stats.messages_forwarded++
			return
		}

		llog('classify', {
			label: classification.label,
			probability: classification.probability.toFixed(4),
			time_ms: (classification.processing_time * 1000).toFixed(1),
			dst: pdu.destination_addr
		}, 'Classification result')

		// 3. Apply confidence threshold
		let label = classification.label
		if (classification.probability < config.classification.confidence_threshold) {
			llog('classify', {probability: classification.probability, threshold: config.classification.confidence_threshold},
				'Below threshold, treating as ham')
			label = 'ham'
		}

		// 4. Apply rule
		let rule = config.classification.rules[label]
		if (!rule) {
			llog('classify', {label: label}, 'No rule for label, defaulting to forward')
			rule = {action: 'forward'}
		}

		if (rule.action == 'forward') {
			await forwardToUpstream(isession, pdu)
			stats.messages_forwarded++

		} else if (rule.action == 'reject') {
			isession.send(pdu.response({
				command_status: rule.error_status || 0x45
			}))
			stats.messages_rejected++
			llog('submit_sm', {label: label, dst: pdu.destination_addr}, 'REJECTED')

		} else if (rule.action == 'drop') {
			// Silent success with synthetic message_id
			isession.send(pdu.response({
				message_id: 'OTS-' + nanoid(8)
			}))
			stats.messages_dropped++
			llog('submit_sm', {label: label, dst: pdu.destination_addr}, 'DROPPED (silent)')
		}
	})
})

// ─────────────────────────────────────────────
// Start server
// ─────────────────────────────────────────────

server.listen({host: config.server.host, port: config.server.port})
llog('startup', {host: config.server.host, port: config.server.port, system_id: config.server.system_id},
	'OTS SMPP server listening')

console.log('OpenTextShield SMPP Interface')
console.log('Server listening on ' + config.server.host + ':' + config.server.port)
console.log('Classification API: ' + apiUrls.map(u => u.href).join(', '))
console.log('PID: ' + process.pid)

// Redirect stdout/stderr to log file now that startup messages are printed
redirectOutputToLog()

// Connect upstream pool
if (config.upstream.host && config.upstream.host != 'upstream-smsc.example.com') {
	createUpstreamPool().then(() => {
		llog('startup', {pool_size: upstream_pool.length}, 'Upstream pool initialized')
		console.log('Upstream pool: ' + upstream_pool.length + ' connections to ' + config.upstream.host + ':' + config.upstream.port)
	}).catch(err => {
		llog('startup', {error: err.message}, 'Upstream pool failed - will retry per-session')
		console.log('WARNING: Upstream pool init failed: ' + err.message)
	})
} else {
	console.log('No upstream configured (using default example host). Messages will be classified but not forwarded.')
	llog('startup', {}, 'No upstream configured - classify-only mode')
}
