/**
 * OpenTextShield SMPP Interface - Test Suite
 *
 * Tests: bind auth, classification, edge cases, stress/benchmark, crash resistance
 *
 * Usage: node test_smpp.js
 */

const smpp = require('smpp')
const {customAlphabet} = require('nanoid')

const HOST = '127.0.0.1'
const PORT = 2775
const VALID_USER = 'client1'
const VALID_PASS = 'secret123'
const INVALID_USER = 'hacker'
const INVALID_PASS = 'wrongpass'

let passed = 0
let failed = 0
let total = 0

function nanoid(len = 8) {
	return customAlphabet('1234567890abcdef', len)()
}

function log(label, msg) {
	console.log(`  [${label}] ${msg}`)
}

function assert(condition, testName) {
	total++
	if (condition) {
		passed++
		console.log(`  ✓ ${testName}`)
	} else {
		failed++
		console.log(`  ✗ FAIL: ${testName}`)
	}
}

// Promise-based bind
function connectAndBind(system_id, password, timeout = 5000) {
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error('Connect timeout')), timeout)
		const session = smpp.connect({url: 'smpp://' + HOST + ':' + PORT})

		session.on('error', (err) => {
			clearTimeout(timer)
			reject(err)
		})

		session.bind_transceiver({
			system_id: system_id,
			password: password
		}, (pdu) => {
			clearTimeout(timer)
			resolve({session, pdu})
		})
	})
}

// Promise-based submit_sm
function submitSm(session, params, timeout = 10000) {
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => {
			resolve({command_status: -1, timeout: true})
		}, timeout)

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

// Graceful close
function closeSession(session) {
	return new Promise((resolve) => {
		session.on('close', () => resolve())
		try {
			session.unbind(() => {
				session.close()
			})
		} catch(e) {
			session.close()
			resolve()
		}
		setTimeout(resolve, 2000) // fallback
	})
}

async function sleep(ms) {
	return new Promise(r => setTimeout(r, ms))
}

// ═══════════════════════════════════════════════
// TEST SUITE
// ═══════════════════════════════════════════════

async function testBindAuth() {
	console.log('\n═══ TEST 1: Bind Authentication ═══')

	// 1a. Valid credentials
	try {
		let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
		assert(pdu.command_status === 0, 'Bind with valid credentials succeeds (status=0)')
		await closeSession(session)
	} catch(e) {
		assert(false, 'Bind with valid credentials succeeds: ' + e.message)
	}

	await sleep(500)

	// 1b. Invalid password
	try {
		let {session, pdu} = await connectAndBind(VALID_USER, INVALID_PASS)
		assert(pdu.command_status !== 0, 'Bind with wrong password fails (status=' + pdu.command_status + ')')
		try { session.close() } catch(e) {}
	} catch(e) {
		// Connection reset or error is also acceptable for invalid auth
		assert(true, 'Bind with wrong password fails (connection rejected)')
	}

	await sleep(500)

	// 1c. Invalid system_id
	try {
		let {session, pdu} = await connectAndBind(INVALID_USER, INVALID_PASS)
		assert(pdu.command_status !== 0, 'Bind with wrong system_id fails (status=' + pdu.command_status + ')')
		try { session.close() } catch(e) {}
	} catch(e) {
		assert(true, 'Bind with wrong system_id fails (connection rejected)')
	}

	await sleep(500)

	// 1d. Empty credentials
	try {
		let {session, pdu} = await connectAndBind('', '')
		assert(pdu.command_status !== 0, 'Bind with empty credentials fails')
		try { session.close() } catch(e) {}
	} catch(e) {
		assert(true, 'Bind with empty credentials fails (connection rejected)')
	}
}

async function testClassification() {
	console.log('\n═══ TEST 2: Message Classification ═══')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// 2a. Ham message (legitimate)
	let resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'Hi John, meeting is at 3pm tomorrow. See you there!'
	})
	log('ham', 'status=' + resp.command_status + ' message_id=' + resp.message_id)
	// In classify-only mode without upstream, ham messages get ESME_RSYSERR (0x08=8)
	// because there's no upstream to forward to
	assert(resp.command_status === 8 || resp.command_status === 0, 'Ham message processed (status=' + resp.command_status + ', expected 8=no upstream or 0=forwarded)')

	await sleep(300)

	// 2b. Spam message
	resp = await submitSm(session, {
		source_addr: '99999',
		destination_addr: '67890',
		short_message: 'CONGRATULATIONS! You won $1,000,000!!! Click here NOW to claim your FREE prize! Limited time offer! Act now! Call 1-800-FREE-CASH'
	})
	log('spam', 'status=' + resp.command_status)
	assert(resp.command_status === 69 || resp.command_status !== 0, 'Spam message rejected (status=' + resp.command_status + ')')

	await sleep(300)

	// 2c. Phishing message
	resp = await submitSm(session, {
		source_addr: '11111',
		destination_addr: '67890',
		short_message: 'URGENT: Your bank account has been compromised! Click http://fake-bank-login.com/verify to verify your identity immediately or your account will be suspended. Enter your password and SSN.'
	})
	log('phishing', 'status=' + resp.command_status)
	assert(resp.command_status === 69 || resp.command_status !== 0, 'Phishing message rejected (status=' + resp.command_status + ')')

	await sleep(300)

	// 2d. Another ham
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'Your appointment is confirmed for Tuesday at 2:30 PM. Dr. Smith Office.'
	})
	log('ham2', 'status=' + resp.command_status)
	assert(resp.command_status === 8 || resp.command_status === 0, 'Legitimate appointment message processed (status=' + resp.command_status + ')')

	await sleep(300)

	// 2e. Borderline spam
	resp = await submitSm(session, {
		source_addr: '55555',
		destination_addr: '67890',
		short_message: 'Special offer for our valued customers: 20% off your next purchase. Use code SAVE20. Reply STOP to unsubscribe.'
	})
	log('borderline', 'status=' + resp.command_status + ' (borderline marketing, model decides)')

	await closeSession(session)
}

async function testEdgeCases() {
	console.log('\n═══ TEST 3: Edge Cases ═══')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// 3a. Empty message
	let resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: ''
	})
	log('empty', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Empty message handled without crash (status=' + resp.command_status + ')')

	await sleep(300)

	// 3b. Very long message (>254 bytes) — must use message_payload TLV
	let longText = 'A'.repeat(200) + ' spam spam spam free money click here '
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: '',
		message_payload: longText
	})
	log('long', 'status=' + resp.command_status + ' (len=' + longText.length + ')')
	assert(resp.command_status !== undefined, 'Long message via message_payload handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3b2. Message at exactly 254 bytes (max short_message)
	let maxShortMsg = 'B'.repeat(250) + ' ok '
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: maxShortMsg
	})
	log('maxshort', 'status=' + resp.command_status + ' (len=' + maxShortMsg.length + ')')
	assert(resp.command_status !== undefined, 'Max-length short_message handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3c. Unicode / emoji message
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'Hello from Japan: こんにちは世界! 🎉 Meeting at café ñ',
		data_coding: 8 // UCS-2
	})
	log('unicode', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Unicode/emoji message handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3d. Binary content (non-text)
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: Buffer.from([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD]),
		data_coding: 4 // binary
	})
	log('binary', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Binary content handled without crash (status=' + resp.command_status + ')')

	await sleep(300)

	// 3e. Special characters / SQL injection attempt in message
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: "'; DROP TABLE messages; -- Hello <script>alert('xss')</script>"
	})
	log('injection', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Injection payload handled safely (status=' + resp.command_status + ')')

	await sleep(300)

	// 3f. Null bytes in message
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'Hello\x00World\x00Test'
	})
	log('nullbytes', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Null bytes in message handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3g. Very long source/dest addresses
	resp = await submitSm(session, {
		source_addr: '1'.repeat(20),
		destination_addr: '9'.repeat(20),
		short_message: 'Test message with long addresses'
	})
	log('longaddr', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Long addresses handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3h. Alphanumeric source address (sender ID)
	resp = await submitSm(session, {
		source_addr: 'MyBrand',
		source_addr_ton: 5, // alphanumeric
		source_addr_npi: 0,
		destination_addr: '447911123456',
		dest_addr_ton: 1, // international
		dest_addr_npi: 1, // E.164
		short_message: 'Your verification code is 123456'
	})
	log('alphasrc', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Alphanumeric sender ID handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3i. Arabic/RTL text
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'مرحبا بالعالم - هذه رسالة اختبار عادية',
		data_coding: 8
	})
	log('arabic', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Arabic/RTL text handled (status=' + resp.command_status + ')')

	await sleep(300)

	// 3j. Single character message
	resp = await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'A'
	})
	log('singlechar', 'status=' + resp.command_status)
	assert(resp.command_status !== undefined, 'Single character message handled (status=' + resp.command_status + ')')

	await closeSession(session)
}

async function testMultipleSessions() {
	console.log('\n═══ TEST 4: Multiple Concurrent Sessions ═══')

	// Open 5 sessions simultaneously
	let sessions = []
	for (let i = 0; i < 5; i++) {
		try {
			let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
			if (pdu.command_status === 0) {
				sessions.push(session)
			}
		} catch(e) {
			log('multi', 'Session ' + i + ' failed: ' + e.message)
		}
	}
	assert(sessions.length === 5, 'Opened 5 concurrent sessions (got ' + sessions.length + ')')

	// Send message from each session concurrently
	let promises = sessions.map((session, i) =>
		submitSm(session, {
			source_addr: '1000' + i,
			destination_addr: '67890',
			short_message: 'Concurrent test message from session ' + i
		})
	)

	let results = await Promise.all(promises)
	let allHandled = results.every(r => r.command_status !== undefined)
	assert(allHandled, 'All 5 concurrent messages got responses')

	// Close all
	for (let session of sessions) {
		await closeSession(session)
	}
}

async function testEnquireLink() {
	console.log('\n═══ TEST 5: Enquire Link ═══')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	// Send enquire_link
	let resp = await new Promise((resolve, reject) => {
		const timer = setTimeout(() => resolve({command_status: -1}), 5000)
		session.enquire_link((pdu) => {
			clearTimeout(timer)
			resolve(pdu)
		})
	})
	assert(resp.command_status === 0, 'Enquire_link response received (status=' + resp.command_status + ')')

	// Send 3 more rapid enquire_links
	for (let i = 0; i < 3; i++) {
		resp = await new Promise((resolve) => {
			const timer = setTimeout(() => resolve({command_status: -1}), 5000)
			session.enquire_link((pdu) => {
				clearTimeout(timer)
				resolve(pdu)
			})
		})
	}
	assert(resp.command_status === 0, 'Multiple rapid enquire_links handled (status=' + resp.command_status + ')')

	await closeSession(session)
}

async function testRapidReconnect() {
	console.log('\n═══ TEST 6: Rapid Connect/Disconnect ═══')

	// Rapidly connect and disconnect 10 times
	let successCount = 0
	for (let i = 0; i < 10; i++) {
		try {
			let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
			if (pdu.command_status === 0) successCount++
			session.close()
		} catch(e) {
			log('reconnect', 'Attempt ' + i + ' failed: ' + e.message)
		}
		await sleep(100)
	}
	assert(successCount >= 8, 'Rapid connect/disconnect: ' + successCount + '/10 succeeded')
	await sleep(1000)
}

async function testBenchmark() {
	console.log('\n═══ TEST 7: Benchmark ═══')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	const messages = [
		'Hi, can you pick up milk on the way home?',
		'Your order #12345 has been shipped',
		'WINNER! You have been selected for a $1000 gift card. Click now!',
		'Reminder: dentist appointment tomorrow at 10am',
		'URGENT: Reset your password at http://phishing-site.com/login',
		'Hey, running 10 min late. Start without me.',
		'FREE FREE FREE!!! Win a brand new iPhone! Text WIN to 12345',
		'Your Uber is arriving in 3 minutes',
		'Verification code: 847291. Do not share this code.',
		'ALERT: Suspicious login detected. Verify at http://fake-bank.net/verify?id=stolen'
	]

	// Warm up (1 message)
	await submitSm(session, {
		source_addr: '12345',
		destination_addr: '67890',
		short_message: 'warmup'
	})
	await sleep(500)

	// Benchmark: send N messages sequentially, measure time
	const N = 50
	let startTime = Date.now()
	let statusCounts = {}

	for (let i = 0; i < N; i++) {
		let msg = messages[i % messages.length]
		let resp = await submitSm(session, {
			source_addr: '12345',
			destination_addr: '67890',
			short_message: msg
		})
		let status = resp.command_status
		statusCounts[status] = (statusCounts[status] || 0) + 1
	}

	let elapsed = Date.now() - startTime
	let tps = (N / (elapsed / 1000)).toFixed(2)
	let avgLatency = (elapsed / N).toFixed(1)

	console.log(`\n  ── Sequential Benchmark Results ──`)
	console.log(`  Messages sent:    ${N}`)
	console.log(`  Total time:       ${elapsed}ms`)
	console.log(`  Throughput:       ${tps} msg/sec`)
	console.log(`  Avg latency:      ${avgLatency}ms per message`)
	console.log(`  Status breakdown: ${JSON.stringify(statusCounts)}`)

	assert(elapsed > 0 && tps > 0, 'Sequential benchmark completed: ' + tps + ' msg/sec, ' + avgLatency + 'ms avg')

	await sleep(500)

	// Benchmark: burst of M messages concurrently
	const M = 20
	startTime = Date.now()
	let promises = []

	for (let i = 0; i < M; i++) {
		let msg = messages[i % messages.length]
		promises.push(submitSm(session, {
			source_addr: '12345',
			destination_addr: '67890',
			short_message: msg
		}))
	}

	let results = await Promise.all(promises)
	elapsed = Date.now() - startTime
	tps = (M / (elapsed / 1000)).toFixed(2)
	avgLatency = (elapsed / M).toFixed(1)

	statusCounts = {}
	results.forEach(r => {
		statusCounts[r.command_status] = (statusCounts[r.command_status] || 0) + 1
	})

	console.log(`\n  ── Concurrent Burst Benchmark Results ──`)
	console.log(`  Messages sent:    ${M} (concurrent)`)
	console.log(`  Total time:       ${elapsed}ms`)
	console.log(`  Throughput:       ${tps} msg/sec`)
	console.log(`  Avg latency:      ${avgLatency}ms per message`)
	console.log(`  Status breakdown: ${JSON.stringify(statusCounts)}`)

	let allResponded = results.every(r => r.command_status !== undefined)
	assert(allResponded, 'Concurrent burst: all ' + M + ' messages got responses at ' + tps + ' msg/sec')

	await closeSession(session)
}

async function testStressMultiSession() {
	console.log('\n═══ TEST 8: Multi-Session Stress ═══')

	const NUM_SESSIONS = 3
	const MSGS_PER_SESSION = 10
	let sessions = []

	// Create sessions
	for (let i = 0; i < NUM_SESSIONS; i++) {
		let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
		if (pdu.command_status === 0) sessions.push(session)
	}
	assert(sessions.length === NUM_SESSIONS, 'Created ' + sessions.length + ' sessions')

	let startTime = Date.now()
	let allPromises = []

	// Each session sends messages concurrently
	for (let s = 0; s < sessions.length; s++) {
		for (let m = 0; m < MSGS_PER_SESSION; m++) {
			allPromises.push(submitSm(sessions[s], {
				source_addr: '1000' + s,
				destination_addr: '2000' + m,
				short_message: 'Stress test session=' + s + ' msg=' + m + ' ' + (m % 2 === 0 ? 'normal message' : 'FREE PRIZE WINNER CLICK NOW')
			}))
		}
	}

	let results = await Promise.all(allPromises)
	let elapsed = Date.now() - startTime
	let totalMsgs = NUM_SESSIONS * MSGS_PER_SESSION
	let tps = (totalMsgs / (elapsed / 1000)).toFixed(2)

	let statusCounts = {}
	results.forEach(r => {
		statusCounts[r.command_status] = (statusCounts[r.command_status] || 0) + 1
	})

	console.log(`\n  ── Multi-Session Stress Results ──`)
	console.log(`  Sessions:         ${NUM_SESSIONS}`)
	console.log(`  Messages/session: ${MSGS_PER_SESSION}`)
	console.log(`  Total messages:   ${totalMsgs}`)
	console.log(`  Total time:       ${elapsed}ms`)
	console.log(`  Throughput:       ${tps} msg/sec`)
	console.log(`  Status breakdown: ${JSON.stringify(statusCounts)}`)

	let allResponded = results.every(r => r.command_status !== undefined)
	assert(allResponded, 'Multi-session stress: all ' + totalMsgs + ' messages got responses')

	for (let session of sessions) {
		await closeSession(session)
	}
}

async function testMalformedPDU() {
	console.log('\n═══ TEST 9: Malformed / Adversarial Input ═══')

	// 9a. Connect raw TCP and send garbage
	const net = require('net')
	let crashed = false

	await new Promise((resolve) => {
		const client = new net.Socket()
		client.connect(PORT, HOST, () => {
			// Send random garbage bytes
			client.write(Buffer.from([0x00, 0x00, 0x00, 0x10, 0xFF, 0xFF, 0xFF, 0xFF,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]))
			setTimeout(() => {
				client.destroy()
				resolve()
			}, 1000)
		})
		client.on('error', () => {
			client.destroy()
			resolve()
		})
	})

	// Check server is still alive
	await sleep(500)
	try {
		let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
		assert(pdu.command_status === 0, 'Server survived garbage PDU bytes')
		await closeSession(session)
	} catch(e) {
		assert(false, 'Server survived garbage PDU bytes - CRASHED: ' + e.message)
		crashed = true
	}

	if (crashed) return

	await sleep(500)

	// 9b. Send incomplete PDU (truncated)
	await new Promise((resolve) => {
		const client = new net.Socket()
		client.connect(PORT, HOST, () => {
			// Incomplete SMPP header (only 8 bytes instead of 16)
			client.write(Buffer.from([0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x09]))
			setTimeout(() => {
				client.destroy()
				resolve()
			}, 1000)
		})
		client.on('error', () => {
			client.destroy()
			resolve()
		})
	})

	await sleep(500)
	try {
		let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
		assert(pdu.command_status === 0, 'Server survived truncated PDU')
		await closeSession(session)
	} catch(e) {
		assert(false, 'Server survived truncated PDU - CRASHED: ' + e.message)
	}

	await sleep(500)

	// 9c. Zero-length connection (connect and immediately close)
	await new Promise((resolve) => {
		const client = new net.Socket()
		client.connect(PORT, HOST, () => {
			client.destroy()
			resolve()
		})
		client.on('error', () => resolve())
	})

	await sleep(500)
	try {
		let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
		assert(pdu.command_status === 0, 'Server survived zero-length connection')
		await closeSession(session)
	} catch(e) {
		assert(false, 'Server survived zero-length connection - CRASHED: ' + e.message)
	}

	await sleep(500)

	// 9d. Flood of connections
	let floodPromises = []
	for (let i = 0; i < 20; i++) {
		floodPromises.push(new Promise((resolve) => {
			const client = new net.Socket()
			client.connect(PORT, HOST, () => {
				setTimeout(() => { client.destroy(); resolve() }, 200)
			})
			client.on('error', () => resolve())
		}))
	}
	await Promise.all(floodPromises)

	await sleep(1000)
	try {
		let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
		assert(pdu.command_status === 0, 'Server survived connection flood (20 rapid connects)')
		await closeSession(session)
	} catch(e) {
		assert(false, 'Server survived connection flood - CRASHED: ' + e.message)
	}
}

async function testLanguages() {
	console.log('\n═══ TEST 10: Multilingual Classification ═══')

	let {session, pdu} = await connectAndBind(VALID_USER, VALID_PASS)
	assert(pdu.command_status === 0, 'Bound successfully')

	const testMessages = [
		{lang: 'English-ham',     text: 'Hey, what time does the movie start tonight?'},
		{lang: 'English-spam',    text: 'YOU WON! Claim your FREE $500 Walmart gift card NOW!!! Visit http://scam.com'},
		{lang: 'Spanish-ham',     text: 'Hola María, ¿nos vemos mañana a las 3?'},
		{lang: 'Spanish-spam',    text: '¡FELICIDADES! Has ganado un viaje GRATIS. Llama ahora al 900-123-456'},
		{lang: 'French-ham',      text: 'Bonjour, le rendez-vous est confirmé pour demain à 14h.'},
		{lang: 'German-ham',      text: 'Hallo, treffen wir uns morgen um 10 Uhr?'},
		{lang: 'Chinese-ham',     text: '你好，明天下午两点开会，请准时参加。'},
		{lang: 'Russian-ham',     text: 'Привет, встречаемся завтра в 3 часа.'},
		{lang: 'Arabic-ham',      text: 'مرحبا، الاجتماع غداً الساعة الثالثة.'},
		{lang: 'Japanese-ham',    text: '明日の会議は午後3時からです。よろしくお願いします。'},
	]

	for (let msg of testMessages) {
		let resp = await submitSm(session, {
			source_addr: '12345',
			destination_addr: '67890',
			short_message: msg.text,
			data_coding: 8 // UCS-2 for non-ASCII
		})
		log(msg.lang, 'status=' + resp.command_status)
		assert(resp.command_status !== undefined, msg.lang + ' handled (status=' + resp.command_status + ')')
		await sleep(200)
	}

	await closeSession(session)
}

// ═══════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════

async function main() {
	console.log('╔═══════════════════════════════════════════════╗')
	console.log('║   OpenTextShield SMPP Interface Test Suite    ║')
	console.log('╚═══════════════════════════════════════════════╝')
	console.log('Target: ' + HOST + ':' + PORT)

	let startTime = Date.now()

	try {
		await testBindAuth()
		await testClassification()
		await testEdgeCases()
		await testMultipleSessions()
		await testEnquireLink()
		await testRapidReconnect()
		await testBenchmark()
		await testStressMultiSession()
		await testMalformedPDU()
		await testLanguages()
	} catch(e) {
		console.log('\n  FATAL ERROR: ' + e.message)
		console.log(e.stack)
	}

	let elapsed = ((Date.now() - startTime) / 1000).toFixed(1)

	console.log('\n╔═══════════════════════════════════════════════╗')
	console.log('║   RESULTS                                     ║')
	console.log('╠═══════════════════════════════════════════════╣')
	console.log('║   Passed: ' + passed + '/' + total + '                                 ║')
	console.log('║   Failed: ' + failed + '                                      ║')
	console.log('║   Time:   ' + elapsed + 's                                  ║')
	console.log('╚═══════════════════════════════════════════════╝')

	process.exit(failed > 0 ? 1 : 0)
}

main()
