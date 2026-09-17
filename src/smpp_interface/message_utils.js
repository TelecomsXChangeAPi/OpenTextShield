/**
 * Pure message helpers for the OTS SMPP proxy.
 *
 * Everything here is free of config loading, sockets and timers, so the proxy
 * and the test suites require the same code. (The tests used to keep hand-made
 * copies of these functions, which could drift from the real proxy.)
 */

const smpp = require('smpp')
const iconv = require('iconv-lite')

// Submit_sm parameters that must round-trip from client to upstream untouched.
// (short_message and message_payload are handled separately because they carry UDH/encoding state.)
const FORWARDED_SUBMIT_SM_PARAMS = [
	'service_type',
	'source_addr_ton', 'source_addr_npi', 'source_addr',
	'dest_addr_ton', 'dest_addr_npi', 'destination_addr',
	'esm_class', 'protocol_id', 'priority_flag',
	'schedule_delivery_time', 'validity_period',
	'registered_delivery', 'replace_if_present_flag',
	'data_coding', 'sm_default_msg_id'
]

const RULE_ACTIONS = ['forward', 'reject', 'drop']
const BELOW_THRESHOLD_ACTIONS = ['use_label', 'forward_as_ham']
const UNCLASSIFIABLE_ACTIONS = ['forward', 'reject']

// The OTS API rejects `text` longer than 512 characters (PredictionRequest).
const DEFAULT_MAX_TEXT_CHARS = 512

// Characters the GSM 03.38 default table can only produce through an escape
// (0x1B) sequence, plus a raw ESC left by an unknown sequence. A default-table
// decode without any of these used no escapes at all.
const GSM_ESCAPED_CHARS = /[\x1B\f^{}\\[~\]|€]/

// data_coding (low 4 bits of DCS, per 3GPP 23.038) → iconv-lite charset name.
// Only the encodings that node-smpp's filters.message.decode does NOT already handle
// need a fallback here — it covers ASCII (0x01), LATIN1 (0x03), UCS-2 (0x08) natively.
//
// We deliberately leave Japanese (0x05 JIS, 0x0A ISO-2022-JP, 0x0D X_0212_1990)
// out: iconv-lite 0.7.x does not register iso-2022-jp, and the spec is ambiguous
// about which JIS encoding form 0x05/0x0D actually map to in practice (some
// operators use Shift_JIS, others EUC-JP). unclassifiableReason() routes those
// to skip-classify instead of producing mojibake for the classifier.
function dataCodingToCharset(dc) {
	if (dc === undefined || dc === null) return null
	switch (dc & 0x0F) {
		case 0x06: return 'iso-8859-5'   // Cyrillic
		case 0x07: return 'iso-8859-8'   // Hebrew
		case 0x08: return 'utf16-be'     // UCS-2 (defensive — should never be hit; node-smpp decodes it)
		case 0x0E: return 'cp949'        // KS C 5601 (Korean) — best-effort
		default:   return null
	}
}

function isGsmDataCoding(dc) {
	const low = (dc || 0) & 0x0F
	return low === 0x00 || low === 0x01
}

// short_message and message_payload as decoded by node-smpp: {message, udh?},
// a bare string, or a Buffer.
function messageParts(pdu) {
	return [pdu.short_message, pdu.message_payload].filter(part => part !== undefined && part !== null)
}

function partMessage(part) {
	return (typeof part === 'object' && !Buffer.isBuffer(part)) ? part.message : part
}

// Printable 7-bit bytes with no ESC. ISO-2022-JP starts in ASCII mode and only
// leaves it through an ESC sequence, so such a payload is plain ASCII text.
function isPlainAscii(buf) {
	return buf.every(b => b === 0x09 || b === 0x0A || b === 0x0D || (b >= 0x20 && b <= 0x7E))
}

// Detect inbound PDUs whose payload the classifier cannot read the way the
// handset will show it. Returns null when the message is classifiable, or a
// short reason string when it should be skipped (see unclassifiable_action).
//
//   - GSM 7-bit national language shift tables (UDH IEI 0x24/0x25) for
//     language codes >= 0x04 (Bengali, Gujarati, Hindi, Kannada, Malayalam,
//     Oriya, Punjabi, Tamil, Telugu, Urdu) require per-language 128-char
//     translation tables that the vendored node-smpp does not ship. A locking
//     shift replaces the whole table, so the default-table decode is wrong.
//     A single shift only changes escaped characters, so when the decoded text
//     shows no escapes the decode is exact and the message is classified:
//     otherwise a sender could add the header to plain Latin text to skip the check.
//   - node-smpp decodes any shift IE with the GSM table whatever the data_coding,
//     so a shift IE on a non-GSM message yields text the handset never shows.
//   - data_coding 0x04 (binary) and 0x09 (pictogram) carry non-textual payloads.
//   - JIS variants (0x05/0x0D) are ambiguous. ISO-2022-JP (0x0A) that is plain
//     ASCII is classified, for the same reason as the single shift above.
function unclassifiableReason(pdu) {
	for (const part of messageParts(pdu)) {
		const udh = Array.isArray(part.udh) ? part.udh : []
		for (const ie of udh) {
			if ((ie[0] !== 0x24 && ie[0] !== 0x25) || ie.length < 3) continue
			const kind = ie[0] === 0x24 ? 'single' : 'locking'
			if (!isGsmDataCoding(pdu.data_coding)) return `national-shift-${kind}-on-non-gsm-data-coding`
			if (ie[2] < 0x04) continue
			const text = partMessage(part)
			if (kind === 'single' && typeof text === 'string' && !GSM_ESCAPED_CHARS.test(text)) continue
			return `national-shift-${kind}-lang-0x${ie[2].toString(16).padStart(2, '0')}`
		}
	}
	const dc = (pdu.data_coding || 0) & 0x0F
	if (dc === 0x04) return 'data-coding-binary'
	if (dc === 0x09) return 'data-coding-pictogram'
	if (dc === 0x05) return 'data-coding-jis-x0208'
	if (dc === 0x0A) {
		const raws = messageParts(pdu).map(partMessage).filter(m => m !== undefined && m !== null && m !== '')
		const plainAscii = raws.length > 0 && raws.every(m => Buffer.isBuffer(m) && isPlainAscii(m))
		if (!plainAscii) return 'data-coding-iso-2022-jp'
	}
	if (dc === 0x0D) return 'data-coding-jis-x0212'
	return null
}

function decodePart(part, dataCoding, log) {
	const raw = partMessage(part)
	if (raw === undefined || raw === null) return ''
	if (typeof raw === 'string') return raw
	if (!Buffer.isBuffer(raw) || raw.length === 0) return ''

	// For data_coding values node-smpp does not register (Cyrillic 0x06, Hebrew 0x07,
	// JIS variants 0x05/0x0A/0x0D, KS C 5601 0x0E, BINARY 0x04, PICTOGRAM 0x09),
	// the decoder leaves .message as a raw Buffer of post-UDH bytes. Naively calling
	// Buffer.toString() on those bytes uses UTF-8 and produces mojibake — so we
	// route through iconv-lite when we can map the data_coding to a known charset.
	const charset = dataCodingToCharset(dataCoding)
	if (charset && iconv.encodingExists(charset)) {
		try {
			return iconv.decode(raw, charset)
		} catch (e) {
			log({error: e.message, charset: charset, dc: dataCoding}, 'iconv decode failed; falling back to utf-8')
		}
	}
	// Last resort — better than dropping the message; classifier may still
	// pick up enough signal from ASCII fragments inside the buffer.
	return raw.toString('utf8')
}

// Extract the human-readable message text for classification.
//
// node-smpp's filters.message.decode already returns a String for ASCII (0x01),
// LATIN1 (0x03), and UCS-2 (0x08) — including UDH-bearing PDUs (UDH is split
// into .udh and the post-UDH bytes are decoded into .message).
//
// Both short_message and message_payload are read. Upstreams receive both, and
// some display message_payload when both are set, so classifying only one would
// let a harmless short_message cover for a phishing message_payload.
//
// This function only affects what goes to the classifier; the original PDU is
// untouched and is forwarded verbatim downstream.
function getMessageText(pdu, log = () => {}) {
	return messageParts(pdu)
		.map(part => decodePart(part, pdu.data_coding, log))
		.filter(text => text !== '')
		.join('\n')
}

// Fit text within the classifier's input limit. A longer message used to be
// rejected by the API and then forwarded unclassified, so padding a phishing
// SMS past the limit skipped the check. Keeps the start and the end, where the
// lure and the link usually are. Counts code points, like the API does, and
// never splits a surrogate pair.
function textForClassification(text, maxChars = DEFAULT_MAX_TEXT_CHARS) {
	const chars = Array.from(text)
	if (chars.length <= maxChars) return text
	const separator = '\n...\n'
	const budget = maxChars - separator.length
	const head = Math.ceil(budget * 0.6)
	return chars.slice(0, head).join('') + separator + chars.slice(chars.length - (budget - head)).join('')
}

// Decide the label to act on. Below confidence_threshold the proxy used to
// forward every spam or phishing verdict as ham, which delivered far more
// phishing than the false blocks it prevented. 'use_label' (the default) acts on
// the model's label; 'forward_as_ham' restores the old behaviour.
function decideLabel(classification, classificationConfig) {
	const threshold = classificationConfig.confidence_threshold
	const belowThreshold = typeof threshold === 'number' && classification.probability < threshold
	const action = classificationConfig.below_threshold_action || 'use_label'
	const label = belowThreshold && action === 'forward_as_ham' ? 'ham' : classification.label
	return {label, belowThreshold, modelLabel: classification.label}
}

// Fail fast on config mistakes that used to surface per message: an unknown
// rule action matched no branch, so the client never got a submit_sm_resp.
function validateClassificationConfig(classificationConfig) {
	const errors = []
	for (const [label, rule] of Object.entries(classificationConfig.rules || {})) {
		if (!rule || !RULE_ACTIONS.includes(rule.action)) {
			errors.push(`rules.${label}.action must be one of ${RULE_ACTIONS.join(', ')}`)
		}
	}
	const optional = [
		['below_threshold_action', BELOW_THRESHOLD_ACTIONS],
		['unclassifiable_action', UNCLASSIFIABLE_ACTIONS],
	]
	for (const [key, allowed] of optional) {
		if (classificationConfig[key] !== undefined && !allowed.includes(classificationConfig[key])) {
			errors.push(`${key} must be one of ${allowed.join(', ')}`)
		}
	}
	const maxChars = classificationConfig.max_text_chars
	if (maxChars !== undefined && !(Number.isInteger(maxChars) && maxChars > 16)) {
		errors.push('max_text_chars must be an integer greater than 16')
	}
	if (errors.length) {
		throw new Error('Invalid classification config: ' + errors.join('; '))
	}
}

// Build the upstream submit_sm by copying every param and TLV the client sent.
//
// Anything we leave out gets silently dropped — and SMPP traffic is full of
// fields that matter even when they look optional: TLV-based concatenation
// (sar_msg_ref_num/sar_total_segments/sar_segment_seqnum), application port
// addressing (source_port/dest_port for WAP push, vCards, MMS notifications),
// validity_period, schedule_delivery_time, payload_type, language_indicator,
// callback_num, etc. The proxy's job is classification, not field surgery, so
// we forward every known param and every registered TLV unchanged.
function buildUpstreamPdu(pdu) {
	let upstream_pdu = {}

	// 1. Standard submit_sm parameters (excluding the message payload itself)
	for (const key of FORWARDED_SUBMIT_SM_PARAMS) {
		if (pdu[key] !== undefined) {
			upstream_pdu[key] = pdu[key]
		}
	}

	// 2. Every TLV the client included. node-smpp tags decoded TLVs by their
	// human name on the pdu object, so iterating its registry is exhaustive.
	// message_payload is handled separately below to keep UDH/encoding logic
	// in one place.
	for (const tag in smpp.tlvs) {
		if (tag === 'message_payload') continue
		if (pdu[tag] !== undefined) {
			upstream_pdu[tag] = pdu[tag]
		}
	}

	// 3. UDH handling. node-smpp's decoder splits UDH into an Array of
	// per-IE Buffers. Its encoder, however, only correctly serializes the
	// FIRST IE when given the array form — multi-IE UDH (e.g. concat header
	// + port addressing, or concat + national language shift) loses everything
	// past udh[0]. Collapse to a single length-prefixed Buffer so the
	// encoder's `else` branch (Buffer.concat) preserves all IEs verbatim.
	if (pdu.short_message && pdu.short_message.udh !== undefined) {
		let udh = pdu.short_message.udh
		if (Array.isArray(udh)) {
			const concatenated = Buffer.concat(udh)
			const len_buf = Buffer.alloc(1)
			len_buf.writeUInt8(concatenated.length, 0)
			udh = Buffer.concat([len_buf, concatenated])
		}
		upstream_pdu.short_message = {udh: udh, message: pdu.short_message.message}
	} else if (pdu.short_message !== undefined) {
		upstream_pdu.short_message = pdu.short_message
	}

	// 4. message_payload — pass through whatever shape the decoder produced.
	// String / {message,udh} / Buffer are all handled by filters.message.encode.
	if (pdu.message_payload !== undefined) {
		upstream_pdu.message_payload = pdu.message_payload
	}

	return upstream_pdu
}

module.exports = {
	FORWARDED_SUBMIT_SM_PARAMS,
	DEFAULT_MAX_TEXT_CHARS,
	dataCodingToCharset,
	unclassifiableReason,
	getMessageText,
	textForClassification,
	decideLabel,
	validateClassificationConfig,
	buildUpstreamPdu,
}
