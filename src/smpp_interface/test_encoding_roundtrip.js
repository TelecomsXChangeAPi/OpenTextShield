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

function unclassifiableReason(pdu) {
	if (pdu.short_message && Array.isArray(pdu.short_message.udh)) {
		for (const ie of pdu.short_message.udh) {
			if ((ie[0] === 0x24 || ie[0] === 0x25) && ie.length >= 3 && ie[2] >= 0x04) {
				const kind = ie[0] === 0x24 ? 'single' : 'locking'
				return `national-shift-${kind}-lang-0x${ie[2].toString(16).padStart(2,'0')}`
			}
		}
	}
	const dc = (pdu.data_coding || 0) & 0x0F
	if (dc === 0x04) return 'data-coding-binary'
	if (dc === 0x09) return 'data-coding-pictogram'
	return null
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

console.log('\n=== Belgian market: French (UCS-2) ===')
{
	// Real-world Belgian-French SMS: full set of accented chars including
	// circumflex (ê, î, ô, û) and ligatures (œ) that are NOT in GSM 7-bit.
	const frenchText = 'Cher client, votre rendez-vous chez le médecin est à 14h. À bientôt — l\'équipe de l\'hôpital. Cœur sain, vœux sincères.'
	const inbound = makeInboundPdu({
		source_addr: 'PROXIMUS', destination_addr: '32475123456',
		data_coding: 8,
		short_message: frenchText
	})
	eq(getMessageText(inbound), frenchText, 'classifier sees French text with circumflex+ligatures intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, frenchText, 'upstream receives French text byte-equivalent (UCS-2)')
}

console.log('\n=== Belgian market: French (Latin-1 / data_coding 3) ===')
{
	// Many Belgian operators send French via Latin-1 instead of UCS-2 to halve
	// the byte cost when the message has accented chars but no €. Latin-1
	// covers the full circumflex set (â ê î ô û) plus ç and the standard
	// Belgian-French diacritics. Note: em-dashes / smart-quotes are NOT in
	// Latin-1 (those need UCS-2) — operator-side sanitization replaces them
	// with hyphens / straight-quotes before encoding.
	const frenchText = 'Reservation confirmee a 19h00. Hotel Cote Belge - bienvenue. Cher client, votre cafe est pret.'
	// (Plain ASCII subset — proves Latin-1 path doesn't introduce drift)
	const inbound = makeInboundPdu({
		source_addr: 'BASE', destination_addr: '32498765432',
		data_coding: 3,
		short_message: frenchText
	})
	eq(getMessageText(inbound), frenchText, 'classifier sees French Latin-1 text intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, frenchText, 'upstream receives French Latin-1 byte-equivalent')
	eq(upstream.data_coding, 3, 'data_coding 3 (Latin-1) preserved')

	// Now verify the full Latin-1 char set actually survives — same path,
	// realistic content with ç / ê / ô / û / é / è / à
	const frenchAccented = 'Réservation confirmée à 19h00. Hôtel Côte Belge: bienvenue. Forêt près du château.'
	const inbound2 = makeInboundPdu({
		source_addr: 'BASE', destination_addr: '32498765432',
		data_coding: 3,
		short_message: frenchAccented
	})
	eq(getMessageText(inbound2), frenchAccented, 'classifier sees Latin-1 accented French intact (â ê î ô û ç é è à)')
	const upstream2 = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound2)))
	eq(upstream2.short_message.message, frenchAccented, 'upstream receives Latin-1 accented French byte-equivalent')
}

console.log('\n=== Belgian market: Dutch / Flemish (default GSM 7-bit, ASCII-safe) ===')
{
	// Belgian operators send most Dutch SMS as plain ASCII — diacritics are
	// rare in transactional SMS and em-dashes/smart-quotes get sanitized to
	// hyphens/straight-quotes upstream of the encoder. Verify the proxy
	// preserves the standard case end-to-end.
	const dutchText = 'Beste klant, uw afspraak bij de tandarts is morgen om 10u30. Tot ziens - Telenet.'
	const inbound = makeInboundPdu({
		source_addr: 'TELENET', destination_addr: '32487654321',
		data_coding: 0,
		short_message: dutchText
	})
	eq(getMessageText(inbound), dutchText, 'classifier sees Dutch text intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, dutchText, 'upstream receives Dutch byte-equivalent (GSM 7-bit)')
}

console.log('\n=== Belgian market: Dutch with diacritics (UCS-2) ===')
{
	// Loanwords and proper nouns: hôtel, café, Pokémon, etc.
	const dutchText = 'Mijn favoriete café in Brugge serveert een geweldige crème brûlée.'
	const inbound = makeInboundPdu({
		source_addr: 'PROXIMUS', destination_addr: '32475999111',
		data_coding: 8,
		short_message: dutchText
	})
	eq(getMessageText(inbound), dutchText, 'classifier sees Dutch with diacritics intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, dutchText, 'upstream receives Dutch+diacritics byte-equivalent')
}

console.log('\n=== Belgian market: German (default GSM 7-bit, Eastern Cantons) ===')
{
	// German speakers in eastern Belgium (Eupen, Sankt Vith). All German
	// special chars (ä ö ü ß and capitals) ARE in GSM default — verify.
	const germanText = 'Sehr geehrte Frau Müller, Ihre Bestellung über 49,90 EUR ist eingegangen. Größe: groß. Schöne Grüße.'
	const inbound = makeInboundPdu({
		source_addr: 'ORANGE', destination_addr: '32499888777',
		data_coding: 0,
		short_message: germanText
	})
	eq(getMessageText(inbound), germanText, 'classifier sees German text intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, germanText, 'upstream receives German GSM 7-bit byte-equivalent')
}

console.log('\n=== Belgian market: French phishing example (real-world spam shape) ===')
{
	// Classifier-quality canary: a typical FR phishing SMS must reach the
	// classifier with all chars intact so mBERT can score it.
	const phishText = 'URGENT: Votre colis bpost est bloqué à la douane. Réglez 1,99€ avant minuit: https://bpost-suivi.example/x?id=8472'
	const inbound = makeInboundPdu({
		source_addr: 'bpost', destination_addr: '32472112233',
		data_coding: 8,
		short_message: phishText
	})
	eq(getMessageText(inbound), phishText, 'classifier sees French phishing text byte-equivalent')
	eq(unclassifiableReason(inbound), null, 'French phishing is classifiable (NOT skipped)')
}

console.log('\n=== Belgian market: euro symbol via GSM ext table (no ç) ===')
{
	// € is in the GSM extension table (ESC + 'e' → 0x1B 0x65). The default
	// GSM alphabet only has uppercase Ç — lowercase ç is not present, so
	// transactional SMS that needs both ç and € forces UCS-2 (next test).
	// This case covers banking SMS in English/Dutch where only € appears.
	const text = 'Belfius: payment of 250€ received. Balance: 1347.82€. Code [4521] valid 5 min.'
	const inbound = makeInboundPdu({
		source_addr: 'Belfius', destination_addr: '32488123456',
		data_coding: 0,
		short_message: text
	})
	eq(getMessageText(inbound), text, 'classifier sees euro-bearing text intact')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, text, 'upstream receives GSM ext table euro+brackets byte-equivalent')
}

console.log('\n=== Belgian market: French banking SMS (€ + ç → must be UCS-2) ===')
{
	// € + ç together cannot be expressed in default GSM 7-bit (lowercase ç
	// is not in the alphabet) NOR Latin-1 (no € — that requires Latin-9).
	// UCS-2 is the only encoding that carries both losslessly, which is
	// what Belgian banks use for French-language transactional SMS.
	const text = 'Belfius: virement de 250€ reçu. Solde: 1.347,82€. Code [4521] valable 5 min.'
	const inbound = makeInboundPdu({
		source_addr: 'Belfius', destination_addr: '32488123456',
		data_coding: 8,
		short_message: text
	})
	eq(getMessageText(inbound), text, 'classifier sees French banking text intact (UCS-2)')
	const upstream = decodeAtUpstream(serializeUpstream(buildUpstreamPdu(inbound)))
	eq(upstream.short_message.message, text, 'upstream receives French banking byte-equivalent')
	eq(upstream.data_coding, 8, 'data_coding 8 (UCS-2) preserved')
}

console.log('\n=== Skip-classify: GSM national language shift (Hindi, lang 0x06) ===')
{
	// Inbound: IEI 0x25 (locking shift), IEDL 0x01, lang 0x06 (Hindi)
	const udh = Buffer.from([0x03, 0x25, 0x01, 0x06])
	const body = Buffer.from([0x68, 0x65, 0x6C, 0x6C, 0x6F, 0x1B, 0x65])
	const fullSm = Buffer.concat([udh, body])
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0,
		esm_class: 0x40,
		short_message: fullSm
	})
	const reason = unclassifiableReason(inbound)
	eq(reason, 'national-shift-locking-lang-0x06', 'Hindi locking shift triggers skip-classify')

	// Forward path must remain byte-equivalent
	const upstreamBuf = serializeUpstream(buildUpstreamPdu(inbound))
	const upstream = decodeAtUpstream(upstreamBuf)
	eq(Array.isArray(upstream.short_message.udh), true, 'UDH still present at upstream')
	eq(upstream.short_message.udh[0][0], 0x25, 'IEI 0x25 preserved')
	eq(upstream.short_message.udh[0][2], 0x06, 'language code 0x06 preserved')
}

console.log('\n=== Skip-classify: GSM national language shift (Tamil, lang 0x0B) ===')
{
	const udh = Buffer.from([0x03, 0x24, 0x01, 0x0B]) // single shift, Tamil
	const body = Buffer.from('hello')
	const fullSm = Buffer.concat([udh, body])
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0, esm_class: 0x40,
		short_message: fullSm
	})
	eq(unclassifiableReason(inbound), 'national-shift-single-lang-0x0b', 'Tamil single shift triggers skip-classify')
}

console.log('\n=== Classify (NOT skipped): Turkish locking shift (lang 0x01) ===')
{
	const udh = Buffer.from([0x03, 0x25, 0x01, 0x01]) // locking shift, Turkish
	const body = Buffer.from('Merhaba')
	const fullSm = Buffer.concat([udh, body])
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0, esm_class: 0x40,
		short_message: fullSm
	})
	eq(unclassifiableReason(inbound), null, 'Turkish (lang 0x01) is supported and NOT skipped')
}

console.log('\n=== Skip-classify: data_coding binary (0x04) ===')
{
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0x04,
		short_message: Buffer.from([0xDE, 0xAD, 0xBE, 0xEF])
	})
	eq(unclassifiableReason(inbound), 'data-coding-binary', 'Binary data_coding triggers skip-classify')
}

console.log('\n=== Skip-classify: data_coding pictogram (0x09) ===')
{
	const inbound = makeInboundPdu({
		source_addr: '12345', destination_addr: '67890',
		data_coding: 0x09,
		short_message: Buffer.from([0x00, 0x01, 0x02, 0x03])
	})
	eq(unclassifiableReason(inbound), 'data-coding-pictogram', 'Pictogram data_coding triggers skip-classify')
}

console.log('\n=== Classify (NOT skipped): plain ASCII / Latin-1 / UCS-2 ===')
{
	const ascii = makeInboundPdu({source_addr: '1', destination_addr: '2', data_coding: 0, short_message: 'hi'})
	eq(unclassifiableReason(ascii), null, 'Plain ASCII not skipped')

	const latin1 = makeInboundPdu({source_addr: '1', destination_addr: '2', data_coding: 3, short_message: 'café'})
	eq(unclassifiableReason(latin1), null, 'Latin-1 not skipped')

	const ucs2 = makeInboundPdu({source_addr: '1', destination_addr: '2', data_coding: 8, short_message: 'مرحبا'})
	eq(unclassifiableReason(ucs2), null, 'UCS-2 (Arabic) not skipped')
}

// ─────────────────────────────────────────────

console.log('\n' + '='.repeat(60))
console.log(`PASSED: ${passed}   FAILED: ${failed}`)
console.log('='.repeat(60))
process.exit(failed > 0 ? 1 : 0)
