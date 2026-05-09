/**
 * Encoding & PDU-transparency round-trip tests for the OTS SMPP proxy.
 *
 * Exercises buildUpstreamPdu() and getMessageText() in isolation — no live
 * proxy/upstream/API needed. Each case constructs an inbound submit_sm PDU
 * the way node-smpp would deliver it after parsing, runs it through the
 * proxy's transformation, then serializes the upstream PDU and compares
 * the on-wire bytes / TLVs / fields against the input.
 *
 * Goal: guarantee transparency across encodings (UCS-2 Arabic / Chinese,
 * GSM 7-bit Turkish via national shift table, Cyrillic ISO-8859-5, etc.)
 * and across optional submit_sm fields and TLVs.
 */

const smpp = require('smpp')
const { PDU } = require('smpp/lib/pdu')
const iconv = require('iconv-lite')
const fs = require('fs')

// The proxy's getMessageText() and buildUpstreamPdu() are vendored verbatim
// below. Keep them in sync with ots_smpp_proxy.js — if they drift, this test
// is the canary. Importing the proxy module directly would boot its TCP
// server and start its upstream-pool reconnect loop, so we copy instead.

const FORWARDED_SUBMIT_SM_PARAMS = [
	'service_type',
	'source_addr_ton', 'source_addr_npi', 'source_addr',
	'dest_addr_ton', 'dest_addr_npi', 'destination_addr',
	'esm_class', 'protocol_id', 'priority_flag',
	'schedule_delivery_time', 'validity_period',
	'registered_delivery', 'replace_if_present_flag',
	'data_coding', 'sm_default_msg_id'
]

function dataCodingToCharset(dc) {
	if (dc === undefined || dc === null) return null
	switch (dc & 0x0F) {
		case 0x06: return 'iso-8859-5'
		case 0x07: return 'iso-8859-8'
		case 0x08: return 'utf16-be'
		case 0x0A: return 'iso-2022-jp'
		case 0x0E: return 'cp949'
		default:   return null
	}
}

function getMessageText(pdu) {
	let raw = null
	if (pdu.short_message != undefined && pdu.short_message.message != undefined) {
		const m = pdu.short_message.message
		if (m !== '' && !(Buffer.isBuffer(m) && m.length === 0)) raw = m
	}
	if (raw == null && ('message_payload' in pdu) && pdu.message_payload != undefined) {
		if (typeof pdu.message_payload === 'object' && pdu.message_payload.message != undefined) {
			const m = pdu.message_payload.message
			if (m !== '' && !(Buffer.isBuffer(m) && m.length === 0)) raw = m
		} else if (typeof pdu.message_payload === 'string' || Buffer.isBuffer(pdu.message_payload)) {
			raw = pdu.message_payload
		}
	}
	if (raw == null) return ''
	if (typeof raw === 'string') return raw
	if (Buffer.isBuffer(raw)) {
		const charset = dataCodingToCharset(pdu.data_coding)
		if (charset && iconv.encodingExists(charset)) {
			try { return iconv.decode(raw, charset) } catch(e) {}
		}
		return raw.toString('utf8')
	}
	return ''
}

function buildUpstreamPdu(pdu) {
	let upstream_pdu = {}
	for (const key of FORWARDED_SUBMIT_SM_PARAMS) {
		if (pdu[key] !== undefined) upstream_pdu[key] = pdu[key]
	}
	for (const tag in smpp.tlvs) {
		if (tag === 'message_payload') continue
		if (pdu[tag] !== undefined) upstream_pdu[tag] = pdu[tag]
	}
	if (pdu.short_message && pdu.short_message.udh !== undefined) {
		let udh = pdu.short_message.udh
		if (Array.isArray(udh)) {
			const concatenated = Buffer.concat(udh)
			const len_buf = Buffer.alloc(1)
			len_buf.writeUInt8(concatenated.length, 0)
			udh = Buffer.concat([len_buf, concatenated])
		}
		upstream_pdu.short_message = { udh: udh, message: pdu.short_message.message }
	} else if (pdu.short_message !== undefined) {
		upstream_pdu.short_message = pdu.short_message
	}
	if (pdu.message_payload !== undefined) upstream_pdu.message_payload = pdu.message_payload
	return upstream_pdu
}

// ─────────────────────────────────────────────
// Test harness
// ─────────────────────────────────────────────

let passed = 0, failed = 0

function eq(actual, expected, name) {
	const a = Buffer.isBuffer(actual) ? actual.toString('hex') : actual
	const e = Buffer.isBuffer(expected) ? expected.toString('hex') : expected
	const ok = (typeof a === 'object' && typeof e === 'object')
		? JSON.stringify(a) === JSON.stringify(e)
		: a === e
	if (ok) {
		console.log(`  ✓ ${name}`)
		passed++
	} else {
		console.log(`  ✗ FAIL ${name}`)
		console.log(`      expected: ${JSON.stringify(e)}`)
		console.log(`      actual:   ${JSON.stringify(a)}`)
		failed++
	}
}

// Build a submit_sm PDU buffer from raw fields (bypasses node-smpp's encode-side
// charset substitution by passing buffers directly), then parse it back via
// PDU.fromBuffer to simulate exactly what an inbound client PDU looks like
// to the proxy.
function makeInboundPdu(fields) {
	const pdu = new PDU('submit_sm', fields)
	const buf = pdu.toBuffer()
	return PDU.fromBuffer(buf)
}

function serializeUpstream(upstream_fields) {
	const pdu = new PDU('submit_sm', upstream_fields)
	return pdu.toBuffer()
}

// Decode the on-wire buffer the upstream would receive, then verify what
// arrives matches what the client sent.
function decodeAtUpstream(buf) {
	return PDU.fromBuffer(buf)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

console.log('\n=== Arabic (UCS-2) ===')
{
	const arabicText = 'مرحبا، الاجتماع غداً الساعة الثالثة.'
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 8,
		short_message: arabicText
	})
	eq(getMessageText(inbound), arabicText, 'classifier sees correct Arabic text')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, arabicText, 'upstream receives Arabic text byte-equivalent')
	eq(upstream.data_coding, 8, 'upstream data_coding preserved (UCS-2)')
}

console.log('\n=== Chinese (UCS-2) ===')
{
	const chineseText = '你好，明天下午两点开会，请准时参加。'
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 8,
		short_message: chineseText
	})
	eq(getMessageText(inbound), chineseText, 'classifier sees correct Chinese text')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, chineseText, 'upstream receives Chinese text byte-equivalent')
}

console.log('\n=== Emoji (UCS-2 with surrogate pairs) ===')
{
	const emojiText = 'Free pizza 🍕 today only! Click 👉 http://bit.ly/x'
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 8,
		short_message: emojiText
	})
	eq(getMessageText(inbound), emojiText, 'classifier sees emoji intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, emojiText, 'upstream receives emoji intact')
}

console.log('\n=== Turkish (GSM 7-bit, basic chars only) ===')
{
	const turkishAscii = 'Merhaba, bugun toplanti var.' // ASCII subset
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0,
		short_message: turkishAscii
	})
	eq(getMessageText(inbound), turkishAscii, 'classifier sees Turkish ASCII text')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, turkishAscii, 'upstream receives Turkish ASCII byte-equivalent')
}

console.log('\n=== Cyrillic (UCS-2 — most common Russian path) ===')
{
	const russianText = 'Здравствуйте, завтра встреча в 15:00.'
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 8,
		short_message: russianText
	})
	eq(getMessageText(inbound), russianText, 'classifier sees Russian text')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, russianText, 'upstream receives Russian byte-equivalent')
}

console.log('\n=== UDH concatenated SMS ===')
{
	// 6-byte UDH header: 05 00 03 AB 02 01 (concat IE: ref=0xAB, total=2, seq=1)
	const udh = Buffer.from([0x05, 0x00, 0x03, 0xAB, 0x02, 0x01])
	const body = Buffer.from('Part 1 of a multipart message that exceeds the single SMS limit and must be split across two PDUs.')
	const fullSm = Buffer.concat([udh, body])
	// Pass as buffer to bypass the encoder; inbound has UDHI bit set (0x40)
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0,
		esm_class: 0x40,
		short_message: fullSm
	})
	const text = getMessageText(inbound)
	eq(text.startsWith('Part 1 of a multipart'), true, 'classifier sees post-UDH text only')
	const upstreamBuf = serializeUpstream(buildUpstreamPdu(inbound))
	const upstream = decodeAtUpstream(upstreamBuf)
	eq(upstream.esm_class & 0x40, 0x40, 'upstream UDHI bit preserved')
	eq(Array.isArray(upstream.short_message.udh), true, 'upstream UDH parsed as array')
	eq(upstream.short_message.udh[0][0], 0x00, 'upstream UDH IEI (0x00 = concat 8-bit) preserved')
	eq(upstream.short_message.udh[0][2], 0xAB, 'upstream UDH concat reference preserved')
	eq(upstream.short_message.udh[0][3], 0x02, 'upstream UDH total parts preserved')
	eq(upstream.short_message.udh[0][4], 0x01, 'upstream UDH seq number preserved')
}

console.log('\n=== Multi-IE UDH (concat + port addressing) ===')
{
	// UDH with TWO IEs: concat (05 00 03 CC 02 01) + port-8bit (04 02 23 F4)
	const udh = Buffer.from([0x0A, // total UDH length = 10
		0x00, 0x03, 0xCC, 0x02, 0x01,           // concat IE
		0x04, 0x02, 0x23, 0xF4                   // application port 8-bit IE
	])
	const body = Buffer.from('Hello')
	const fullSm = Buffer.concat([udh, body])
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0,
		esm_class: 0x40,
		short_message: fullSm
	})
	const upstreamBuf = serializeUpstream(buildUpstreamPdu(inbound))
	const upstream = decodeAtUpstream(upstreamBuf)
	eq(upstream.short_message.udh.length, 2, 'upstream preserves BOTH UDH IEs (regression: array form drops past [0])')
	eq(upstream.short_message.udh[0][0], 0x00, 'first IE = concat header')
	eq(upstream.short_message.udh[1][0], 0x04, 'second IE = port-8bit')
}

console.log('\n=== TLV preservation (sar_msg_ref_num, source_port, dest_port) ===')
{
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0,
		short_message: 'TLV test',
		sar_msg_ref_num: 0x1234,
		sar_total_segments: 3,
		sar_segment_seqnum: 2,
		source_port: 5000,
		dest_port: 9200,
		user_message_reference: 42,
		payload_type: 1,
		privacy_indicator: 2,
		language_indicator: 5
	})
	const upstreamBuf = serializeUpstream(buildUpstreamPdu(inbound))
	const upstream = decodeAtUpstream(upstreamBuf)
	eq(upstream.sar_msg_ref_num, 0x1234, 'sar_msg_ref_num preserved')
	eq(upstream.sar_total_segments, 3, 'sar_total_segments preserved')
	eq(upstream.sar_segment_seqnum, 2, 'sar_segment_seqnum preserved')
	eq(upstream.source_port, 5000, 'source_port preserved (regression: WAP push)')
	eq(upstream.dest_port, 9200, 'dest_port preserved (regression: WAP push)')
	eq(upstream.user_message_reference, 42, 'user_message_reference preserved')
	eq(upstream.payload_type, 1, 'payload_type preserved')
	eq(upstream.privacy_indicator, 2, 'privacy_indicator preserved')
	eq(upstream.language_indicator, 5, 'language_indicator preserved')
}

console.log('\n=== Submit_sm optional params (validity, schedule, priority) ===')
{
	const inbound = makeInboundPdu({
		service_type: 'CMT',
		source_addr_ton: 5, source_addr_npi: 0,
		source_addr: 'BRAND',
		dest_addr_ton: 1, dest_addr_npi: 1,
		destination_addr: '447700123456',
		data_coding: 0,
		short_message: 'Validity test',
		priority_flag: 2,
		validity_period: '000000003600000R',     // 1 hour relative
		schedule_delivery_time: '000000010000000R', // 1 day relative
		replace_if_present_flag: 1,
		sm_default_msg_id: 7,
		protocol_id: 0x40,
		registered_delivery: 1
	})
	const upstreamBuf = serializeUpstream(buildUpstreamPdu(inbound))
	const upstream = decodeAtUpstream(upstreamBuf)
	eq(upstream.service_type, 'CMT', 'service_type preserved')
	eq(upstream.source_addr_ton, 5, 'source_addr_ton preserved')
	eq(upstream.source_addr, 'BRAND', 'source_addr (alphanumeric sender) preserved')
	eq(upstream.priority_flag, 2, 'priority_flag preserved')
	eq(upstream.replace_if_present_flag, 1, 'replace_if_present_flag preserved')
	eq(upstream.sm_default_msg_id, 7, 'sm_default_msg_id preserved')
	eq(upstream.protocol_id, 0x40, 'protocol_id preserved (regression: SMS replace)')
	eq(upstream.registered_delivery, 1, 'registered_delivery preserved')
}

console.log('\n=== Long message via message_payload TLV ===')
{
	const longText = 'A'.repeat(300) // exceeds short_message 254-byte limit
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0,
		message_payload: longText
	})
	eq(getMessageText(inbound), longText, 'classifier sees full message_payload text')
	const upstreamBuf = serializeUpstream(buildUpstreamPdu(inbound))
	const upstream = decodeAtUpstream(upstreamBuf)
	eq(upstream.message_payload.message, longText, 'upstream receives full message_payload')
}

// ─────────────────────────────────────────────

console.log('\n' + '='.repeat(60))
console.log(`PASSED: ${passed}   FAILED: ${failed}`)
console.log('='.repeat(60))
process.exit(failed > 0 ? 1 : 0)
