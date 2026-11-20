/**
 * The shared conformance fixtures (docs/contract.md section 11) against the TS port.
 *
 *   node --experimental-strip-types --test euler_metric_naming/contract/ts/conformance.test.ts
 */
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';

import {
	CanonicalizationError,
	canonicalize,
	compare,
	compareFiles,
	definitionKey,
	definitionKeyObject,
	evaluateCondition,
	fileEvidence,
	formatPercent,
	formatSig3,
	globMatch,
	hashValue,
	legacyTable,
	loadEvalFile,
	manifestHash,
	manifestIds,
	measure,
	normalize,
	phrase,
	recomputeCell,
	relevantKeys,
	sha256Hex,
	utf8Encode
} from './euler-contract.ts';
import type { EvalFile } from './euler-contract.ts';
import { CLOSED_SETS, VOCABULARIES } from './euler-contract.data.ts';
import { nodeCompareFiles, nodeSha256Hex } from './euler-contract-node.ts';

const FIXTURES = new URL('../fixtures/', import.meta.url);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function load(name: string): Any {
	return JSON.parse(readFileSync(new URL(name, FIXTURES), 'utf8'));
}

const files = new Map<string, EvalFile>();
function evalFile(name: string): EvalFile {
	let file = files.get(name);
	if (file === undefined) {
		file = loadEvalFile(load(`evals/${name}`));
		files.set(name, file);
	}
	return file;
}

// -- jcs ------------------------------------------------------------------------------------

const SPECIAL: { [name: string]: number } = { NaN: Number.NaN, Infinity: Infinity, '-Infinity': -Infinity };

function parseInput(text: string): unknown {
	// integer literals are integer-typed (BigInt), as Python's int
	return /^-?\d+$/.test(text) ? BigInt(text) : JSON.parse(text);
}

test('jcs', () => {
	for (const testCase of load('jcs.json').cases) {
		const value =
			'special' in testCase
				? SPECIAL[testCase.special]
				: 'inputText' in testCase
					? parseInput(testCase.inputText)
					: testCase.input;
		if (testCase.error) {
			assert.throws(() => canonicalize(value), CanonicalizationError, testCase.name);
		} else {
			assert.equal(canonicalize(value), testCase.canonical, testCase.name);
		}
	}
});

// -- hashes ---------------------------------------------------------------------------------

test('sha256 matches node:crypto', () => {
	for (const size of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 4096]) {
		const bytes = new Uint8Array(randomBytes(size));
		assert.equal(sha256Hex(bytes), nodeSha256Hex(bytes), `size ${size}`);
	}
	const text = 'café 😀 ﬁ  ';
	assert.deepEqual([...utf8Encode(text)], [...Buffer.from(text, 'utf8')]);
	assert.equal(sha256Hex(utf8Encode('')), createHash('sha256').update('').digest('hex'));
});

const HASHES = load('hashes.json');

test('hash values', () => {
	for (const testCase of HASHES.values) {
		assert.equal(canonicalize(testCase.input), testCase.canonical, testCase.name);
		assert.equal(hashValue(testCase.input), testCase.hash, testCase.name);
	}
	for (const testCase of HASHES.referenceIdentities) {
		assert.equal(canonicalize(testCase.input), testCase.canonical, testCase.name);
		assert.equal(hashValue(testCase.input), testCase.hash, testCase.name);
	}
});

test('manifests', () => {
	for (const testCase of HASHES.manifests) {
		assert.equal(canonicalize(manifestIds(testCase.ids)), testCase.canonical, testCase.name);
		assert.equal(manifestHash(testCase.ids), testCase.hash, testCase.name);
	}
	const d1 = load('evals/d1.json');
	const population = HASHES.manifests.find((c: Any) => c.name === 'the d1 population');
	assert.equal(d1.protocol.population.manifest, population.hash);
});

test('definition keys', () => {
	for (const testCase of HASHES.definitionKeys) {
		for (const name of testCase.files) {
			const contract = evalFile(name).contract;
			assert.deepEqual(definitionKeyObject(contract), testCase.object, `${testCase.name} ${name}`);
			assert.equal(canonicalize(definitionKeyObject(contract)), testCase.canonical, testCase.name);
			assert.equal(definitionKey(contract), testCase.hash, testCase.name);
		}
	}
});

// -- conditions, globs, measures ----------------------------------------------------------------

test('conditions', () => {
	const fixture = load('conditions.json');
	const truth: { [name: string]: boolean | null } = { true: true, false: false, unknown: null };
	for (const testCase of fixture.cases) {
		const result = evaluateCondition(testCase.condition, fixture.parameters, fixture.observations);
		assert.equal(result, truth[testCase.expected], testCase.name);
	}
});

test('globs', () => {
	for (const testCase of load('globs.json').cases) {
		assert.equal(globMatch(testCase.pattern, testCase.name), testCase.match, `${testCase.pattern} | ${testCase.name}`);
	}
});

test('measures and phrases', () => {
	const fixture = load('measures.json');
	for (const testCase of fixture.measures) {
		const d = measure(testCase.kind, testCase.a, testCase.b);
		if (testCase.d === null) {
			assert.equal(d, null, testCase.name);
			continue;
		}
		assert.ok(d !== null && Math.abs(d - testCase.d) <= 5e-7, `${testCase.name}: ${d}`);
		assert.equal(phrase(testCase.kind, 'L', testCase.a, testCase.b, d as number), testCase.phrase, testCase.name);
	}
	for (const testCase of fixture.percent) assert.equal(formatPercent(testCase.d), testCase.text, String(testCase.d));
	for (const testCase of fixture.sig3) assert.equal(formatSig3(testCase.value), testCase.text, String(testCase.value));
});

// -- legacy -------------------------------------------------------------------------------------

test('legacy adapter', () => {
	const table = legacyTable();
	for (const testCase of load('legacy.json').cases) {
		const contract: Any = normalize(testCase.protocol, testCase.metricSet, testCase.meta);
		const descriptors = contract.descriptors;
		const reducers = contract.reducers;
		delete contract.descriptors;
		delete contract.reducers;
		assert.deepEqual(contract, testCase.expected, testCase.name);
		assert.deepEqual(Object.keys(descriptors).sort(), testCase.descriptorKeys, testCase.name);
		for (const [key, descriptor] of Object.entries(descriptors)) {
			assert.deepEqual(descriptor, table.descriptors[key], `${testCase.name} ${key}`);
		}
		assert.deepEqual(reducers, table.reducers, testCase.name);
	}
});

// -- verdicts -----------------------------------------------------------------------------------

const VERDICTS = load('verdicts.json').cases;

test('verdicts', () => {
	for (const testCase of VERDICTS) {
		const fileA = evalFile(testCase.a);
		const fileB = evalFile(testCase.b);
		const verdict = compareFiles(fileA, fileB, testCase.metric);
		const name = testCase.name;
		assert.equal(verdict.tier, testCase.tier, name);
		assert.equal(verdict.label, testCase.label, name);
		assert.deepEqual(
			verdict.reasons.map((reason) => reason.code),
			testCase.codes,
			name
		);
		assert.equal(verdict.shared, testCase.shared, name);
		if ('measures' in testCase) {
			const measures = verdict.reasons.filter((reason) => reason.measure !== undefined);
			assert.deepEqual(
				measures.map((reason) => [reason.key, reason.measure?.kind]),
				testCase.measures.map((m: Any) => [m.key, m.kind]),
				name
			);
			measures.forEach((reason, index) => {
				assert.ok(Math.abs((reason.measure?.d as number) - testCase.measures[index].d) <= 5e-7, name);
			});
		}
		if ('hint' in testCase) assert.equal(verdict.reasons[0]?.hint, testCase.hint, name);
		if ('recompute' in testCase) {
			const itemsA = fileEvidence(fileA).items as Set<string>;
			const itemsB = fileEvidence(fileB).items as Set<string>;
			const shared = [...itemsA].filter((item) => itemsB.has(item));
			const gotA = recomputeCell(fileA.contract, fileA, testCase.metric, shared) as number;
			const gotB = recomputeCell(fileB.contract, fileB, testCase.metric, shared) as number;
			assert.ok(Math.abs(gotA - testCase.recompute.a) <= 1e-9, `${name}: ${gotA}`);
			assert.ok(Math.abs(gotB - testCase.recompute.b) <= 1e-9, `${name}: ${gotB}`);
		}
		const viaNode = nodeCompareFiles(fileA, fileB, testCase.metric);
		assert.deepEqual(viaNode, verdict, `${name} (node:crypto)`);
	}
});

test('verdicts are symmetric in tier, codes and shared', () => {
	for (const testCase of VERDICTS) {
		const forward = compareFiles(evalFile(testCase.a), evalFile(testCase.b), testCase.metric);
		const backward = compareFiles(evalFile(testCase.b), evalFile(testCase.a), testCase.metric);
		assert.equal(backward.tier, forward.tier, testCase.name);
		assert.deepEqual(
			backward.reasons.map((r) => r.code),
			forward.reasons.map((r) => r.code),
			testCase.name
		);
		assert.equal(backward.shared, forward.shared, testCase.name);
	}
});

// -- reference identity schemes and relevance ------------------------------------------------------

test('reference identity schemes', () => {
	const fixture = load('references.json');
	const contract = (reference: Any) =>
		normalize({ ...structuredClone(fixture.template), references: { gt: reference } }, fixture.metricSet);
	for (const testCase of fixture.cases) {
		for (const [a, b] of [
			[testCase.a, testCase.b],
			[testCase.b, testCase.a]
		]) {
			const verdict = compare(contract(a), contract(b), fixture.metric);
			assert.equal(verdict.tier, testCase.tier, testCase.name);
			assert.equal(verdict.label, testCase.label, testCase.name);
			assert.deepEqual(
				verdict.reasons.map((reason) => reason.code),
				testCase.codes,
				testCase.name
			);
			assert.equal(verdict.reasons[0]?.hint ?? null, testCase.hint, testCase.name);
		}
	}
});

test('relevant keys', () => {
	for (const testCase of load('relevance.json').cases) {
		const a = evalFile(testCase.a).contract;
		const b = testCase.b ? evalFile(testCase.b).contract : null;
		assert.deepEqual(relevantKeys(a, b, testCase.metric), testCase.keys, testCase.name);
		assert.deepEqual(relevantKeys(b ?? a, b === null ? null : a, testCase.metric), testCase.keys, testCase.name);
	}
});

// -- generated data -------------------------------------------------------------------------------

test('generated data matches the JSON sources', () => {
	const ids = readdirSync(new URL('../vocabularies/', import.meta.url))
		.filter((name) => name.endsWith('.json'))
		.map((name) => name.slice(0, -'.json'.length))
		.sort();
	assert.deepEqual(Object.keys(VOCABULARIES).sort(), ids);
	for (const id of ids) {
		const document = JSON.parse(readFileSync(new URL(`../vocabularies/${id}.json`, import.meta.url), 'utf8'));
		assert.deepEqual(VOCABULARIES[id], document, id);
	}
	const legacy = JSON.parse(readFileSync(new URL('../legacy/protocol-1.json', import.meta.url), 'utf8'));
	assert.deepEqual(legacyTable(), legacy);
	assert.deepEqual([...CLOSED_SETS.tiers], ['same', 'paired', 'comparable', 'incomparable']);
});
