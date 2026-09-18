/**
 * Unit tests for message_utils.js: the classification bypass fixes and the
 * threshold decision. No live proxy, upstream or API needed.
 *
 *   node test_message_utils.js
 */

const fs = require('fs')
const path = require('path')
const { PDU } = require('smpp/lib/pdu')
const {
	unclassifiableReason,
	getMessageText,
	textForClassification,
	decideLabel,
	validateClassificationConfig,
} = require('./message_utils')

let passed = 0, failed = 0

function eq(actual, expected, name) {
	if (JSON.stringify(actual) === JSON.stringify(expected)) {
		console.log(`  ✓ ${name}`)
		passed++
	} else {
		console.log(`  ✗ FAIL ${name}`)
		console.log(`      expected: ${JSON.stringify(expected)}`)
		console.log(`      actual:   ${JSON.stringify(actual)}`)
		failed++
	}
}

function throwsWith(fn, fragment, name) {
	try {
		fn()
		eq('no error', `error containing "${fragment}"`, name)
	} catch (e) {
		eq(e.message.includes(fragment), true, name)
	}
}

// Parse a submit_sm the way the proxy receives it from a client.
function inbound(fields) {
	return PDU.fromBuffer(new PDU('submit_sm', {source_addr: '1', destination_addr: '2', ...fields}).toBuffer())
}

console.log('\n=== Shift header on a non-GSM message is not trusted ===')
{
	// node-smpp decodes any shift IE with the GSM table, so the classifier would
	// see garbage while the handset shows the real UCS-2 text.
	const udh = Buffer.from([0x03, 0x24, 0x01, 0x01]) // single shift, Turkish
	const ucs2 = Buffer.from('Your card is blocked, call 0800 123', 'utf16le').swap16()
	const pdu = inbound({data_coding: 0x08, esm_class: 0x40, short_message: Buffer.concat([udh, ucs2])})
	eq(unclassifiableReason(pdu), 'national-shift-single-on-non-gsm-data-coding', 'shift IE on UCS-2 is flagged')
}

console.log('\n=== ISO-2022-JP (0x0A) that is plain ASCII is classified ===')
{
	const pdu = inbound({data_coding: 0x0A, short_message: Buffer.from('Your parcel is held. Pay the fee at http://parcel-fee.example')})
	eq(unclassifiableReason(pdu), null, 'plain ASCII under ISO-2022-JP is not skipped')
	eq(getMessageText(pdu), 'Your parcel is held. Pay the fee at http://parcel-fee.example', 'classifier sees the ASCII text')

	const japanese = inbound({data_coding: 0x0A, short_message: Buffer.from([0x1B, 0x24, 0x42, 0x24, 0x33, 0x1B, 0x28, 0x42])})
	eq(unclassifiableReason(japanese), 'data-coding-iso-2022-jp', 'real ISO-2022-JP with escapes is still skipped')
}

console.log('\n=== JIS X 0208 / X 0212 stay skipped (encoding form is ambiguous) ===')
{
	eq(unclassifiableReason(inbound({data_coding: 0x05, short_message: Buffer.from('hello')})), 'data-coding-jis-x0208', '0x05 skipped')
	eq(unclassifiableReason(inbound({data_coding: 0x0D, short_message: Buffer.from('hello')})), 'data-coding-jis-x0212', '0x0D skipped')
}

console.log('\n=== Both short_message and message_payload are classified ===')
{
	const pdu = inbound({
		data_coding: 0,
		short_message: 'Hi, see below',
		message_payload: 'Your parcel is held. Pay the 1.99 fee at http://parcel-fee.example',
	})
	eq(getMessageText(pdu), 'Hi, see below\nYour parcel is held. Pay the 1.99 fee at http://parcel-fee.example',
		'classifier sees both fields, so a harmless short_message cannot hide the payload')

	const payloadOnly = inbound({data_coding: 0, message_payload: 'Long message in the payload'})
	eq(getMessageText(payloadOnly), 'Long message in the payload', 'payload-only message unchanged')
}

console.log('\n=== textForClassification keeps long messages within the API limit ===')
{
	eq(textForClassification('short message'), 'short message', 'short text unchanged')

	const exact = 'x'.repeat(512)
	eq(textForClassification(exact), exact, 'text at the limit unchanged')

	const long = 'Dear customer, ' + 'lorem ipsum dolor sit amet '.repeat(40) + 'verify at http://bank-verify.example/login'
	const fitted = textForClassification(long)
	eq(Array.from(fitted).length <= 512, true, 'long text fits the 512 character API limit')
	eq(fitted.startsWith('Dear customer, '), true, 'start of the message kept')
	eq(fitted.endsWith('http://bank-verify.example/login'), true, 'link at the end kept')

	const emoji = '🔒'.repeat(600)
	const fittedEmoji = textForClassification(emoji)
	eq(Array.from(fittedEmoji).length <= 512, true, 'emoji text counted by code point')
	eq(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(fittedEmoji), false,
		'no surrogate pair split')

	eq(Array.from(textForClassification(long, 100)).length <= 100, true, 'custom max_text_chars respected')
}

console.log('\n=== decideLabel: confidence threshold ===')
{
	const cfg = {confidence_threshold: 0.7}
	eq(decideLabel({label: 'phishing', probability: 0.55}, cfg),
		{label: 'phishing', belowThreshold: true, modelLabel: 'phishing'},
		'default acts on an unsure phishing verdict instead of forwarding it as ham')
	eq(decideLabel({label: 'phishing', probability: 0.55}, {...cfg, below_threshold_action: 'forward_as_ham'}),
		{label: 'ham', belowThreshold: true, modelLabel: 'phishing'},
		'forward_as_ham restores the old behaviour')
	eq(decideLabel({label: 'spam', probability: 0.93}, {...cfg, below_threshold_action: 'forward_as_ham'}),
		{label: 'spam', belowThreshold: false, modelLabel: 'spam'},
		'confident verdicts are never changed')
	eq(decideLabel({label: 'ham', probability: 0.4}, cfg),
		{label: 'ham', belowThreshold: true, modelLabel: 'ham'},
		'unsure ham stays ham')
	eq(decideLabel({label: 'spam', probability: 0.2}, {}).label, 'spam', 'no threshold configured')
}

console.log('\n=== validateClassificationConfig ===')
{
	const shipped = JSON.parse(fs.readFileSync(path.join(__dirname, 'config.json'), 'utf8'))
	let ok = true
	try { validateClassificationConfig(shipped.classification) } catch (e) { ok = e.message }
	eq(ok, true, 'shipped config.json is valid')

	const base = {rules: {ham: {action: 'forward'}, spam: {action: 'reject'}, phishing: {action: 'drop'}}}
	throwsWith(() => validateClassificationConfig({rules: {spam: {action: 'quarantine'}}}),
		'rules.spam.action', 'unknown rule action is rejected at startup')
	throwsWith(() => validateClassificationConfig({...base, below_threshold_action: 'block'}),
		'below_threshold_action', 'unknown below_threshold_action is rejected')
	throwsWith(() => validateClassificationConfig({...base, unclassifiable_action: 'drop'}),
		'unclassifiable_action', 'unknown unclassifiable_action is rejected')
	throwsWith(() => validateClassificationConfig({...base, max_text_chars: 5}),
		'max_text_chars', 'too small max_text_chars is rejected')
}

console.log('\n' + '='.repeat(60))
console.log(`PASSED: ${passed}   FAILED: ${failed}`)
console.log('='.repeat(60))
process.exit(failed > 0 ? 1 : 0)
