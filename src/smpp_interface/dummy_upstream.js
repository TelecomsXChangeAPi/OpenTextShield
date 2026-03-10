/**
 * Dummy Upstream SMSC
 *
 * Simulates an upstream SMSC that accepts binds, receives submit_sm,
 * returns message_ids, and sends DLRs back after a configurable delay.
 *
 * Usage: node dummy_upstream.js [port]
 * Default port: 2776
 */

const smpp = require('smpp')
const {customAlphabet} = require('nanoid')

const PORT = process.argv[2] || 2776
const SYSTEM_ID = 'DUMMY_SMSC'
const ALLOWED_USER = 'ots_esme'
const ALLOWED_PASS = 'upstream_pass'
const DLR_DELAY_MS = 500  // send DLR 500ms after accepting message
const SIMULATE_DLR = true

let msgCounter = 0
let sessions = {}

function nanoid(len = 8) {
	return customAlphabet('1234567890ABCDEF', len)()
}

function log(func, msg) {
	let ts = new Date().toJSON().slice(0, 23).replace('T', ' ')
	console.log(`${ts} [${func}] ${msg}`)
}

// Stats
let stats = {
	binds: 0,
	messages_received: 0,
	dlrs_sent: 0,
	active_sessions: 0
}

let server = smpp.createServer(function(session) {

	session.on('bind_transceiver', function(pdu) {
		let system_id = pdu.system_id
		let password = pdu.password

		if (system_id == ALLOWED_USER && password == ALLOWED_PASS) {
			session.send(pdu.response({system_id: SYSTEM_ID}))
			let nick = 'DUM-' + nanoid(8)
			session.nickname = nick
			sessions[nick] = session
			stats.binds++
			stats.active_sessions++
			log('bind', 'ACCEPTED ' + system_id + ' as ' + nick)
		} else {
			session.send(pdu.response({command_status: smpp.ESME_RBINDFAIL}))
			log('bind', 'REJECTED ' + system_id)
			session.close()
		}
	})

	session.on('enquire_link', function(pdu) {
		session.send(pdu.response())
	})

	session.on('unbind', function(pdu) {
		log('unbind', 'Session ' + (session.nickname || 'unknown'))
		session.send(pdu.response())
		session.close()
	})

	session.on('close', function() {
		if (session.nickname) {
			delete sessions[session.nickname]
			stats.active_sessions = Math.max(0, stats.active_sessions - 1)
		}
	})

	session.on('error', function(err) {
		log('error', 'Session error: ' + ((err && err.message) ? err.message : err))
		if (session.nickname) {
			delete sessions[session.nickname]
			stats.active_sessions = Math.max(0, stats.active_sessions - 1)
		}
	})

	session.on('submit_sm', function(pdu) {
		msgCounter++
		let message_id = nanoid(8)
		let src = pdu.source_addr
		let dst = pdu.destination_addr

		// Extract text for logging
		let text = ''
		if (pdu.short_message) {
			if (pdu.short_message.message) {
				text = Buffer.isBuffer(pdu.short_message.message) ?
					pdu.short_message.message.toString().substring(0, 40) :
					(pdu.short_message.message + '').substring(0, 40)
			}
		}

		log('submit_sm', '#' + msgCounter + ' ' + src + '->' + dst +
			' msg_id=' + message_id + ' text="' + text + '"')

		// Return success with message_id
		session.send(pdu.response({message_id: message_id}))
		stats.messages_received++

		// Send DLR after delay (if registered_delivery is set and SIMULATE_DLR)
		if (SIMULATE_DLR && pdu.registered_delivery) {
			setTimeout(function() {
				try {
					let dlr_text = 'id:' + message_id + ' sub:001 dlvrd:001 submit date:' +
						formatSmppDate() + ' done date:' + formatSmppDate() +
						' stat:DELIVRD err:000 text:' + (text || '').substring(0, 20)

					session.deliver_sm({
						source_addr: dst,
						destination_addr: src,
						esm_class: 4, // DLR
						short_message: dlr_text
					}, function(resp_pdu) {
						log('dlr', 'DLR sent for msg_id=' + message_id + ' status=' + (resp_pdu ? resp_pdu.command_status : 'unknown'))
						stats.dlrs_sent++
					})
				} catch(e) {
					log('dlr', 'DLR send failed for msg_id=' + message_id + ': ' + e.message)
				}
			}, DLR_DELAY_MS)
		}
	})
})

function formatSmppDate() {
	let d = new Date()
	let yy = (d.getFullYear() % 100).toString().padStart(2, '0')
	let MM = (d.getMonth() + 1).toString().padStart(2, '0')
	let dd = d.getDate().toString().padStart(2, '0')
	let hh = d.getHours().toString().padStart(2, '0')
	let mm = d.getMinutes().toString().padStart(2, '0')
	return yy + MM + dd + hh + mm
}

server.listen({host: '0.0.0.0', port: PORT})

console.log('╔═══════════════════════════════════════════════╗')
console.log('║   Dummy Upstream SMSC                         ║')
console.log('╠═══════════════════════════════════════════════╣')
console.log('║   Port:      ' + PORT + '                                ║')
console.log('║   System ID: ' + SYSTEM_ID + '                        ║')
console.log('║   Login:     ' + ALLOWED_USER + '                          ║')
console.log('║   DLR delay: ' + DLR_DELAY_MS + 'ms                              ║')
console.log('╚═══════════════════════════════════════════════╝')

// Log stats periodically
setInterval(function() {
	log('stats', JSON.stringify(stats))
}, 30000)

process.on('uncaughtException', function(err) {
	console.error('Uncaught:', (err && err.stack) ? err.stack : err)
})
