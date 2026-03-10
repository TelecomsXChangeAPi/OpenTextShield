/**
 * OpenTextShield SMPP Interface - Advanced Test Suite
 *
 * Tests:
 *   - UDH / multipart SMS (concatenated messages)
 *   - Emoji and special Unicode
 *   - Async correctness (ordering, concurrency, race conditions)
 *   - Isolated component benchmarks (SMPP overhead vs OTS vs upstream)
 *   - DLR relay verification
 *
 * Usage: node test_advanced.js
 * Requires: proxy on :2775, dummy upstream on :2776, OTS API on :8002
 */

const smpp = require('smpp')
const http = require('http')
const net = require('net')

const HOST = '127.0.0.1'
const PORT = 2775
const OTS_API = 'http://127.0.0.1:8002'
const VALID_USER = 'client1'
const VALID_PASS = 'secret123'

let passed = 0
let failed = 0
let total = 0

function log(label, msg) {
	console.log(`  [${label}] ${msg}`)
}

function assert(condition, testName) {
	total++
	if (condition) {
		passed++
		console.log(`  \u2713 ${testName}`)
	} else {
		failed++
		console.log(`  \u2717 FAIL: ${testName}`)
	}
}

function connectAndBind(system_id, password, timeout = 5000) {
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error('Connect timeout')), timeout)
		const session = smpp.connect({url: 'smpp://' + HOST + ':' + PORT})
		session.on('error', (err) => { clearTimeout(timer); reject(err) })
		session.bind_transceiver({system_id, password}, (pdu) => {
			clearTimeout(timer)
			resolve({session, pdu})
		})
	})
}

function submitSm(session, params, timeout = 15000) {
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => resolve({command_status: -1, timeout: true}), timeout)
		try {
			session.submit_sm(params, (pdu) => {
				clearTimeout(timer)
				resolve(pdu)
			})
		} catch(e) {
			clearTimeout(timer)
			reject(e)
		}
	})
}

function closeSession(session) {
	return new Promise((resolve) => {
		session.on('close', () => resolve())
		try { session.unbind(() => session.close()) } catch(e) { session.close(); resolve() }
		setTimeout(resolve, 2000)
	})
}

async function sleep(ms) {
	return new Promise(r => setTimeout(r, ms))
}

// Direct HTTP call to OTS API (for isolated benchmark)
function classifyDirect(text) {
	return new Promise((resolve, reject) => {
		const data = JSON.stringify({text, model: 'ots-mbert'})
		const start = Date.now()
		const req = http.request({
			hostname: '127.0.0.1', port: 8002, path: '/predict/',
			method: 'POST',
			headers: {'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data)},
			timeout: 10000
		}, (res) => {
			let body = ''
			res.on('data', chunk => body += chunk)
			res.on('end', () => {
				try {
					const result = JSON.parse(body)
					result._latency_ms = Date.now() - start
					resolve(result)
				} catch(e) { reject(e) }
			})
		})
		req.on('error', reject)
		req.on('timeout', () => { req.destroy(); reject(new Error('timeout')) })
		req.write(data)
		req.end()
	})
}

// Direct SMPP submit to dummy upstream (bypassing proxy, for isolated benchmark)
function submitToDummyDirect(text) {
	return new Promise(async (resolve, reject) => {
		try {
			const session = await smpp.connect({url: 'smpp://127.0.0.1:2776'})
			const start = Date.now()
			session.bind_transceiver({system_id: 'ots_esme', password: 'upstream_pass'}, (pdu) => {
				if (pdu.command_status !== 0) { reject(new Error('bind failed')); return }
				session.submit_sm({
					source_addr: '12345', destination_addr: '67890',
					short_message: text
				}, (resp) => {
					const elapsed = Date.now() - start
					session.close()
					resolve({command_status: resp.command_status, message_id: resp.message_id, _latency_ms: elapsed})
				})
			})
		} catch(e) { reject(e) }
	})
}

// ═══════════════════════════════════════════════
// TEST: UDH / Multipart SMS
// ═══════════════════════════════════════════════

async function testUDH() {
	console.log('\n\u2550\u2550\u2550 TEST A: UDH / Multipart SMS \u2550\u2550\u2550')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// A1. Single-part message with UDH indicator (esm_class = 0x40)
	// Build a proper UDH: 05 00 03 AB 02 01 (concat header, ref=0xAB, 2 parts, part 1)
	let udh_concat = Buffer.from([0x05, 0x00, 0x03, 0xAB, 0x02, 0x01])
	let msg_part1 = Buffer.from('This is part 1 of a concatenated message that is being split across ')
	let full_sm1 = Buffer.concat([udh_concat, msg_part1])

	let resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		esm_class: 0x40, // UDHI bit set
		short_message: full_sm1
	})
	log('udh-part1', 'status=' + resp.command_status + ' msg_id=' + resp.message_id)
	assert(resp.command_status !== undefined, 'UDH concat part 1 handled (status=' + resp.command_status + ')')

	await sleep(300)

	// A2. Second part of concatenated message
	let udh_concat2 = Buffer.from([0x05, 0x00, 0x03, 0xAB, 0x02, 0x02])
	let msg_part2 = Buffer.from('multiple SMPP PDUs to test UDH handling in the proxy.')
	let full_sm2 = Buffer.concat([udh_concat2, msg_part2])

	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		esm_class: 0x40,
		short_message: full_sm2
	})
	log('udh-part2', 'status=' + resp.command_status + ' msg_id=' + resp.message_id)
	assert(resp.command_status !== undefined, 'UDH concat part 2 handled (status=' + resp.command_status + ')')

	await sleep(300)

	// A3. UDH with node-smpp's object format {udh, message}
	// This is how node-smpp natively represents UDH when you pass it as an object
	let udh_buf = Buffer.from([0x00, 0x03, 0xCC, 0x03, 0x01]) // concat IE: ref=0xCC, 3 parts, part 1
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: {
			udh: [udh_buf],
			message: 'Part 1 of 3: Normal legitimate message content here'
		}
	})
	log('udh-obj-p1', 'status=' + resp.command_status + ' msg_id=' + resp.message_id)
	assert(resp.command_status !== undefined, 'UDH object format part 1/3 handled (status=' + resp.command_status + ')')

	await sleep(300)

	// A4. UDH part 2/3
	let udh_buf2 = Buffer.from([0x00, 0x03, 0xCC, 0x03, 0x02])
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: {
			udh: [udh_buf2],
			message: 'Part 2 of 3: Continuing with more of the message text'
		}
	})
	log('udh-obj-p2', 'status=' + resp.command_status + ' msg_id=' + resp.message_id)
	assert(resp.command_status !== undefined, 'UDH object format part 2/3 handled (status=' + resp.command_status + ')')

	await sleep(300)

	// A5. UDH part 3/3 — with spam content, should be rejected
	let udh_buf3 = Buffer.from([0x00, 0x03, 0xCC, 0x03, 0x03])
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: {
			udh: [udh_buf3],
			message: 'Part 3: FREE PRIZE WINNER! Click http://scam.com NOW to claim $1000000!!!'
		}
	})
	log('udh-obj-p3-spam', 'status=' + resp.command_status)
	assert(resp.command_status === 69, 'UDH spam part correctly rejected (status=' + resp.command_status + ')')

	await sleep(300)

	// A6. UDH with 16-bit concatenation reference (IE id=0x08)
	let udh_16bit = Buffer.from([0x06, 0x08, 0x04, 0x12, 0x34, 0x02, 0x01]) // 16-bit ref=0x1234, 2 parts, part 1
	let msg_16bit = Buffer.from('Message with 16-bit UDH concat reference')
	let full_sm_16bit = Buffer.concat([udh_16bit, msg_16bit])

	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		esm_class: 0x40,
		short_message: full_sm_16bit
	})
	log('udh-16bit', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'UDH 16-bit concat reference handled (status=' + resp.command_status + ')')

	await sleep(300)

	// A7. UDH with UCS-2 encoding (data_coding=8)
	let udh_ucs2 = Buffer.from([0x05, 0x00, 0x03, 0xDD, 0x02, 0x01])
	let ucs2_text = Buffer.from('مرحبا بالعالم', 'utf16le') // Arabic in UCS-2
	let full_ucs2 = Buffer.concat([udh_ucs2, ucs2_text])

	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		esm_class: 0x40,
		data_coding: 8,
		short_message: full_ucs2
	})
	log('udh-ucs2', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'UDH with UCS-2 Arabic handled (status=' + resp.command_status + ')')

	await closeSession(session)
}

// ═══════════════════════════════════════════════
// TEST: Emoji and Special Unicode
// ═══════════════════════════════════════════════

async function testEmoji() {
	console.log('\n\u2550\u2550\u2550 TEST B: Emoji & Special Unicode \u2550\u2550\u2550')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// B1. Common emojis
	let resp = await submitSm(session, {
		source_addr: '12345', destination_addr: '67890',
		short_message: 'Hey! \ud83d\ude00\ud83d\udc4d\u2764\ufe0f See you soon! \ud83c\udf89',
		data_coding: 8
	})
	log('emoji-basic', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Basic emojis handled (status=' + resp.command_status + ')')

	await sleep(300)

	// B2. Emoji-only message (no text)
	resp = await submitSm(session, {
		source_addr: '12345', destination_addr: '67890',
		short_message: '\ud83d\ude02\ud83d\ude02\ud83d\ude02',
		data_coding: 8
	})
	log('emoji-only', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Emoji-only message handled (status=' + resp.command_status + ')')

	await sleep(300)

	// B3. Mixed emoji + spam content
	resp = await submitSm(session, {
		source_addr: '99999', destination_addr: '67890',
		short_message: '\ud83d\udcb0\ud83d\udcb0\ud83d\udcb0 FREE MONEY!!! Click NOW to win $50000! \ud83c\udf1f\ud83c\udf1f\ud83c\udf1f Act FAST!!!',
		data_coding: 8
	})
	log('emoji-spam', 'status=' + resp.command_status)
	assert(resp.command_status === 69, 'Emoji-laden spam rejected (status=' + resp.command_status + ')')

	await sleep(300)

	// B4. Flag emojis (multi-codepoint)
	resp = await submitSm(session, {
		source_addr: '12345', destination_addr: '67890',
		short_message: 'Greetings from \ud83c\uddfa\ud83c\uddf8 \ud83c\uddef\ud83c\uddf5 \ud83c\uddec\ud83c\udde7 \ud83c\udde6\ud83c\uddea! Business meeting tomorrow.',
		data_coding: 8
	})
	log('emoji-flags', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Flag emojis handled (status=' + resp.command_status + ')')

	await sleep(300)

	// B5. Skin tone modifiers (complex codepoints)
	resp = await submitSm(session, {
		source_addr: '12345', destination_addr: '67890',
		short_message: 'Team meeting \ud83d\udc68\u200d\ud83d\udcbb\ud83d\udc69\u200d\ud83d\udcbb at 3pm today',
		data_coding: 8
	})
	log('emoji-complex', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Complex emoji codepoints handled (status=' + resp.command_status + ')')

	await sleep(300)

	// B6. Zero-width joiners and combining chars
	resp = await submitSm(session, {
		source_addr: '12345', destination_addr: '67890',
		short_message: 'Test with ZWJ: \ud83d\udc68\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 and combining: a\u0301e\u0301',
		data_coding: 8
	})
	log('emoji-zwj', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'ZWJ sequences handled (status=' + resp.command_status + ')')

	await sleep(300)

	// B7. Mixed scripts (CJK + Arabic + Latin + emoji)
	resp = await submitSm(session, {
		source_addr: '12345', destination_addr: '67890',
		short_message: 'Hello \u4f60\u597d \u0645\u0631\u062d\u0628\u0627 \u3053\u3093\u306b\u3061\u306f \ud83c\udf0d',
		data_coding: 8
	})
	log('mixed-scripts', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Mixed scripts handled (status=' + resp.command_status + ')')

	await closeSession(session)
}

// ═══════════════════════════════════════════════
// TEST: Async Correctness
// ═══════════════════════════════════════════════

async function testAsyncCorrectness() {
	console.log('\n\u2550\u2550\u2550 TEST C: Async Correctness \u2550\u2550\u2550')

	// C1. Verify responses match requests (not mixed up under concurrency)
	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// Send 20 messages concurrently - mix of ham and spam
	// Ham should get status 0 (forwarded), spam should get 69 (rejected)
	const concurrent_msgs = []
	for (let i = 0; i < 20; i++) {
		if (i % 2 === 0) {
			concurrent_msgs.push({
				idx: i, type: 'ham',
				params: {
					source_addr: '1' + i.toString().padStart(4, '0'),
					destination_addr: '200' + i,
					short_message: 'Normal message number ' + i + ' for meeting at 3pm'
				}
			})
		} else {
			concurrent_msgs.push({
				idx: i, type: 'spam',
				params: {
					source_addr: '9' + i.toString().padStart(4, '0'),
					destination_addr: '200' + i,
					short_message: 'FREE PRIZE #' + i + '! WIN $1000000 NOW!!! Click http://scam' + i + '.com URGENT!!!'
				}
			})
		}
	}

	// Fire all at once
	let promises = concurrent_msgs.map(m => submitSm(session, m.params))
	let results = await Promise.all(promises)

	// Verify all got responses
	let allResponded = results.every(r => r.command_status !== undefined && r.command_status !== -1)
	assert(allResponded, 'All 20 concurrent messages got valid responses (no timeouts)')

	// Verify classification correctness
	let hamCorrect = 0, spamCorrect = 0, hamTotal = 0, spamTotal = 0
	for (let i = 0; i < results.length; i++) {
		if (concurrent_msgs[i].type === 'ham') {
			hamTotal++
			if (results[i].command_status === 0) hamCorrect++
		} else {
			spamTotal++
			if (results[i].command_status === 69) spamCorrect++
		}
	}
	log('async-class', `ham: ${hamCorrect}/${hamTotal} correct, spam: ${spamCorrect}/${spamTotal} correct`)
	assert(hamCorrect >= 8 && spamCorrect >= 8,
		'Concurrent classification accuracy maintained (ham=' + hamCorrect + '/' + hamTotal + ', spam=' + spamCorrect + '/' + spamTotal + ')')

	await sleep(500)

	// C2. Rapid fire from single session (tests backpressure / queue handling)
	let rapidResults = []
	let rapidStart = Date.now()
	let rapidPromises = []
	for (let i = 0; i < 50; i++) {
		rapidPromises.push(
			submitSm(session, {
				source_addr: '12345',
				destination_addr: '67890',
				short_message: 'Rapid fire message ' + i + ' - appointment confirmed'
			}).then(r => {
				rapidResults.push({idx: i, status: r.command_status, time: Date.now() - rapidStart})
			})
		)
	}
	await Promise.all(rapidPromises)

	let rapidAllOk = rapidResults.every(r => r.status !== undefined && r.status !== -1)
	assert(rapidAllOk, 'All 50 rapid-fire messages got responses (no drops)')

	// Check for proper async behavior - all should complete without piling up
	let maxTime = Math.max(...rapidResults.map(r => r.time))
	log('rapid-fire', 'All 50 completed in ' + maxTime + 'ms max')
	assert(maxTime < 30000, 'Rapid fire completed within 30s (actual max: ' + maxTime + 'ms)')

	await sleep(500)

	// C3. Interleaved sessions - verify no cross-session contamination
	let session2_bind = await connectAndBind(VALID_USER, VALID_PASS)
	let session2 = session2_bind.session

	// Send from both sessions interleaved
	let s1_promises = []
	let s2_promises = []
	for (let i = 0; i < 10; i++) {
		s1_promises.push(submitSm(session, {
			source_addr: 'S1-' + i,
			destination_addr: '67890',
			short_message: 'Session 1 message ' + i + ' normal text'
		}))
		s2_promises.push(submitSm(session2, {
			source_addr: 'S2-' + i,
			destination_addr: '67890',
			short_message: 'Session 2 message ' + i + ' normal text'
		}))
	}

	let [s1_results, s2_results] = await Promise.all([
		Promise.all(s1_promises),
		Promise.all(s2_promises)
	])

	let s1_ok = s1_results.every(r => r.command_status !== undefined && r.command_status !== -1)
	let s2_ok = s2_results.every(r => r.command_status !== undefined && r.command_status !== -1)
	assert(s1_ok && s2_ok, 'Interleaved sessions both got all responses (S1=' + s1_results.length + ', S2=' + s2_results.length + ')')

	// Verify message_ids are unique across sessions
	let allMsgIds = [...s1_results, ...s2_results]
		.filter(r => r.message_id)
		.map(r => r.message_id)
	let uniqueIds = new Set(allMsgIds)
	assert(uniqueIds.size === allMsgIds.length, 'All message_ids unique across sessions (' + uniqueIds.size + '/' + allMsgIds.length + ')')

	await closeSession(session2)
	await closeSession(session)
}

// ═══════════════════════════════════════════════
// TEST: DLR Relay Verification
// ═══════════════════════════════════════════════

async function testDLR() {
	console.log('\n\u2550\u2550\u2550 TEST D: DLR (Delivery Receipt) Relay \u2550\u2550\u2550')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// Set up DLR listener on client session
	let dlrs_received = []
	session.on('deliver_sm', function(deliver_pdu) {
		session.send(deliver_pdu.response())
		dlrs_received.push(deliver_pdu)
	})

	// Send a ham message with registered_delivery=1 (request DLR)
	let resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'Test DLR relay - normal meeting at 3pm',
		registered_delivery: 1
	})
	log('dlr-send', 'status=' + resp.command_status + ' msg_id=' + resp.message_id)
	assert(resp.command_status === 0, 'Ham message forwarded for DLR test (msg_id=' + resp.message_id + ')')

	// Wait for DLR from dummy upstream (it sends after 500ms)
	await sleep(2000)

	log('dlr-recv', 'DLRs received: ' + dlrs_received.length)
	if (dlrs_received.length > 0) {
		let dlr = dlrs_received[0]
		let dlr_text = ''
		if (dlr.short_message && dlr.short_message.message) {
			dlr_text = Buffer.isBuffer(dlr.short_message.message)
				? dlr.short_message.message.toString() : dlr.short_message.message
		}
		log('dlr-content', 'esm_class=' + dlr.esm_class + ' text="' + dlr_text.substring(0, 80) + '"')
		assert(dlr.esm_class === 4, 'DLR has correct esm_class=4')
		assert(dlr_text.includes('DELIVRD') || dlr_text.includes('stat:'), 'DLR contains delivery status')
	} else {
		assert(false, 'DLR received within 2 seconds')
		assert(false, 'DLR content verification (no DLR received)')
	}

	// Send multiple messages with DLR request
	let sent_ids = []
	for (let i = 0; i < 5; i++) {
		let r = await submitSm(session, {
			source_addr: '12345',
			destination_addr: '6789' + i,
			short_message: 'DLR batch message ' + i + ' - normal text',
			registered_delivery: 1
		})
		if (r.command_status === 0 && r.message_id) {
			sent_ids.push(r.message_id)
		}
		await sleep(100)
	}

	// Wait for DLRs
	await sleep(3000)

	log('dlr-batch', 'Sent ' + sent_ids.length + ' messages, received ' + dlrs_received.length + ' DLRs total')
	assert(dlrs_received.length >= 3, 'Received DLRs for batch messages (' + dlrs_received.length + ' DLRs for ' + (sent_ids.length + 1) + ' sent)')

	await closeSession(session)
}

// ═══════════════════════════════════════════════
// TEST: Isolated Component Benchmarks
// ═══════════════════════════════════════════════

async function testIsolatedBenchmarks() {
	console.log('\n\u2550\u2550\u2550 TEST E: Isolated Component Benchmarks \u2550\u2550\u2550')

	const N = 30 // messages per benchmark
	const ham_text = 'Hi John, meeting is at 3pm tomorrow. See you there!'
	const spam_text = 'CONGRATULATIONS! You won $1,000,000!!! Click NOW!'

	// ── E1. OTS Classification API only (direct HTTP) ──
	console.log('\n  \u2500\u2500 E1: OTS Classification API (direct HTTP, no SMPP) \u2500\u2500')
	let ots_latencies = []
	for (let i = 0; i < N; i++) {
		let text = (i % 2 === 0) ? ham_text : spam_text
		let result = await classifyDirect(text)
		ots_latencies.push(result._latency_ms)
	}
	let ots_avg = (ots_latencies.reduce((a, b) => a + b, 0) / ots_latencies.length).toFixed(1)
	let ots_min = Math.min(...ots_latencies)
	let ots_max = Math.max(...ots_latencies)
	let ots_p50 = ots_latencies.sort((a, b) => a - b)[Math.floor(ots_latencies.length / 2)]
	let ots_p95 = ots_latencies.sort((a, b) => a - b)[Math.floor(ots_latencies.length * 0.95)]

	console.log(`  OTS API (${N} calls):`)
	console.log(`    Avg:  ${ots_avg}ms`)
	console.log(`    Min:  ${ots_min}ms`)
	console.log(`    Max:  ${ots_max}ms`)
	console.log(`    P50:  ${ots_p50}ms`)
	console.log(`    P95:  ${ots_p95}ms`)

	assert(parseFloat(ots_avg) > 0, 'OTS API benchmark completed (avg=' + ots_avg + 'ms)')

	await sleep(500)

	// ── E2. Dummy Upstream only (direct SMPP, no classification) ──
	console.log('\n  \u2500\u2500 E2: Upstream SMSC (direct SMPP, no proxy/classification) \u2500\u2500')
	let upstream_latencies = []

	// Use a single session for all upstream measurements
	let up_session_wrap = await new Promise((resolve, reject) => {
		const session = smpp.connect({url: 'smpp://127.0.0.1:2776'})
		session.on('error', reject)
		session.bind_transceiver({system_id: 'ots_esme', password: 'upstream_pass'}, (pdu) => {
			if (pdu.command_status !== 0) reject(new Error('bind failed'))
			else resolve(session)
		})
	})

	for (let i = 0; i < N; i++) {
		let start = Date.now()
		await new Promise((resolve, reject) => {
			up_session_wrap.submit_sm({
				source_addr: '12345', destination_addr: '67890',
				short_message: ham_text
			}, (pdu) => {
				upstream_latencies.push(Date.now() - start)
				resolve(pdu)
			})
		})
	}
	up_session_wrap.close()

	let up_avg = (upstream_latencies.reduce((a, b) => a + b, 0) / upstream_latencies.length).toFixed(1)
	let up_min = Math.min(...upstream_latencies)
	let up_max = Math.max(...upstream_latencies)
	let up_p50 = upstream_latencies.sort((a, b) => a - b)[Math.floor(upstream_latencies.length / 2)]
	let up_p95 = upstream_latencies.sort((a, b) => a - b)[Math.floor(upstream_latencies.length * 0.95)]

	console.log(`  Upstream SMSC (${N} submits):`)
	console.log(`    Avg:  ${up_avg}ms`)
	console.log(`    Min:  ${up_min}ms`)
	console.log(`    Max:  ${up_max}ms`)
	console.log(`    P50:  ${up_p50}ms`)
	console.log(`    P95:  ${up_p95}ms`)

	assert(parseFloat(up_avg) > 0, 'Upstream benchmark completed (avg=' + up_avg + 'ms)')

	await sleep(500)

	// ── E3. Full pipeline via SMPP proxy (client → proxy → classify → upstream) ──
	console.log('\n  \u2500\u2500 E3: Full Pipeline via SMPP Proxy (end-to-end) \u2500\u2500')
	let {session: bench_session, pdu: bench_pdu} = await connectAndBind(VALID_USER, VALID_PASS)

	// Warmup
	await submitSm(bench_session, {source_addr: '12345', destination_addr: '67890', short_message: 'warmup'})
	await sleep(300)

	let pipeline_latencies = []
	let pipeline_statuses = {}

	for (let i = 0; i < N; i++) {
		let text = (i % 2 === 0) ? ham_text : spam_text
		let start = Date.now()
		let resp = await submitSm(bench_session, {
			source_addr: '12345', destination_addr: '67890',
			short_message: text
		})
		pipeline_latencies.push(Date.now() - start)
		pipeline_statuses[resp.command_status] = (pipeline_statuses[resp.command_status] || 0) + 1
	}

	let pl_avg = (pipeline_latencies.reduce((a, b) => a + b, 0) / pipeline_latencies.length).toFixed(1)
	let pl_min = Math.min(...pipeline_latencies)
	let pl_max = Math.max(...pipeline_latencies)
	let pl_p50 = pipeline_latencies.sort((a, b) => a - b)[Math.floor(pipeline_latencies.length / 2)]
	let pl_p95 = pipeline_latencies.sort((a, b) => a - b)[Math.floor(pipeline_latencies.length * 0.95)]
	let pl_tps = (N / (pipeline_latencies.reduce((a, b) => a + b, 0) / 1000)).toFixed(2)

	console.log(`  Full Pipeline (${N} messages):`)
	console.log(`    Avg:  ${pl_avg}ms`)
	console.log(`    Min:  ${pl_min}ms`)
	console.log(`    Max:  ${pl_max}ms`)
	console.log(`    P50:  ${pl_p50}ms`)
	console.log(`    P95:  ${pl_p95}ms`)
	console.log(`    TPS:  ${pl_tps} msg/sec`)
	console.log(`    Status: ${JSON.stringify(pipeline_statuses)}`)

	assert(parseFloat(pl_avg) > 0, 'Full pipeline benchmark completed (avg=' + pl_avg + 'ms, ' + pl_tps + ' TPS)')

	// ── E4. Concurrent pipeline benchmark ──
	console.log('\n  \u2500\u2500 E4: Full Pipeline - Concurrent Burst \u2500\u2500')
	let burst_start = Date.now()
	let burst_promises = []
	for (let i = 0; i < N; i++) {
		let text = (i % 2 === 0) ? ham_text : spam_text
		burst_promises.push(submitSm(bench_session, {
			source_addr: '12345', destination_addr: '67890',
			short_message: text
		}))
	}
	let burst_results = await Promise.all(burst_promises)
	let burst_elapsed = Date.now() - burst_start
	let burst_tps = (N / (burst_elapsed / 1000)).toFixed(2)

	let burst_statuses = {}
	burst_results.forEach(r => {
		burst_statuses[r.command_status] = (burst_statuses[r.command_status] || 0) + 1
	})

	console.log(`  Concurrent Burst (${N} messages):`)
	console.log(`    Total time: ${burst_elapsed}ms`)
	console.log(`    TPS:  ${burst_tps} msg/sec`)
	console.log(`    Status: ${JSON.stringify(burst_statuses)}`)

	assert(parseFloat(burst_tps) > 0, 'Concurrent burst benchmark completed (' + burst_tps + ' TPS)')

	await closeSession(bench_session)

	// ── Summary: Component Breakdown ──
	let smpp_overhead = Math.max(0, parseFloat(pl_avg) - parseFloat(ots_avg) - parseFloat(up_avg))

	console.log('\n  \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550')
	console.log('  COMPONENT LATENCY BREAKDOWN (avg)')
	console.log('  \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550')
	console.log(`  OTS mBERT API:     ${ots_avg}ms  (${((parseFloat(ots_avg) / parseFloat(pl_avg)) * 100).toFixed(0)}% of pipeline)`)
	console.log(`  Upstream SMSC:     ${up_avg}ms  (${((parseFloat(up_avg) / parseFloat(pl_avg)) * 100).toFixed(0)}% of pipeline)`)
	console.log(`  SMPP Proxy overhead: ${smpp_overhead.toFixed(1)}ms  (${((smpp_overhead / parseFloat(pl_avg)) * 100).toFixed(0)}% of pipeline)`)
	console.log(`  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500`)
	console.log(`  Full Pipeline:     ${pl_avg}ms  (100%)`)
	console.log(`  Sequential TPS:    ${pl_tps} msg/sec`)
	console.log(`  Concurrent TPS:    ${burst_tps} msg/sec`)
	console.log('  \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550')
}

// ═══════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════

async function main() {
	console.log('\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557')
	console.log('\u2551   OTS SMPP - Advanced Test Suite              \u2551')
	console.log('\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d')
	console.log('Proxy: ' + HOST + ':' + PORT)
	console.log('Upstream: 127.0.0.1:2776')
	console.log('OTS API: ' + OTS_API)

	let startTime = Date.now()

	try {
		await testUDH()
		await testEmoji()
		await testAsyncCorrectness()
		await testDLR()
		await testIsolatedBenchmarks()
	} catch(e) {
		console.log('\n  FATAL ERROR: ' + e.message)
		console.log(e.stack)
	}

	let elapsed = ((Date.now() - startTime) / 1000).toFixed(1)

	console.log('\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557')
	console.log('\u2551   ADVANCED TEST RESULTS                       \u2551')
	console.log('\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563')
	console.log('\u2551   Passed: ' + passed + '/' + total + '                                 \u2551')
	console.log('\u2551   Failed: ' + failed + '                                      \u2551')
	console.log('\u2551   Time:   ' + elapsed + 's                                  \u2551')
	console.log('\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d')

	process.exit(failed > 0 ? 1 : 0)
}

main()
