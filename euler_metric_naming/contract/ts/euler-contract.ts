/**
 * The evaluation protocol contract, schema 2 (docs/contract.md of
 * euler-metric-naming) for TypeScript consumers.
 *
 * A line-by-line port of the Python package `euler_metric_naming.contract`:
 * the same fixtures (contract/fixtures) give the same results in both. No
 * dependencies, no DOM or Node types; the only relative import is the
 * generated data file. Uses only syntax that Node's type stripping and
 * `erasableSyntaxOnly` accept (no enums, namespaces or parameter properties).
 *
 * Consumers: normalize any results file with `loadEvalFile` (or `normalize`)
 * and judge one metric cell of two evaluations with `compareFiles` / `compare`.
 */
import { CLOSED_SETS, LEGACY_PROTOCOL_1 } from './euler-contract.data.ts';

// -- types ---------------------------------------------------------------------------

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type Role = 'reference' | 'definition' | 'population' | 'domain' | 'derived' | 'provenance';
export type CompareRule = 'equal' | 'measure' | 'pair' | 'ignore';
export type ObserveKind = 'region' | 'count' | 'fraction' | 'factor' | 'size';
export type DisplayScale = 'relative' | 'complement' | 'absolute' | 'none';
export type Direction = 'lower' | 'higher' | 'none';
export type MetricKind = 'score' | 'statistic' | 'observation';
export type RecomputeOp = 'mean' | 'median' | 'pool' | 'none';
export type PoolForm = 'mean' | 'rms' | 'none';
export type AxisRoleName = 'space' | 'reduction' | 'stratum' | 'variant' | 'family' | 'stage' | 'kind';
export type ConditionOp = 'exists' | 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge' | 'in';
export type Tier = 'same' | 'paired' | 'comparable' | 'incomparable';
export type ReasonCode =
	| 'schema'
	| 'reference'
	| 'revision'
	| 'not_recorded'
	| 'stratification'
	| 'parameter'
	| 'domain'
	| 'population_disjoint'
	| 'population_paired'
	| 'population_partial'
	| 'population_unknown';
/** Three-valued truth: `null` is unknown. */
export type Truth = boolean | null;

export type Condition =
	| { all: Condition[] }
	| { any: Condition[] }
	| { not: Condition }
	| { param: string; op: ConditionOp; value?: JsonValue }
	| { observation: string; op: ConditionOp; value?: JsonValue };

export interface Affects {
	metrics?: string[];
	axes?: { [role: string]: string[] };
}

export interface Descriptor {
	label?: string;
	role: Role | string;
	compare?: CompareRule | string;
	observe?: ObserveKind | string;
	unit?: string | null;
	affects?: Affects;
	appliesWhen?: Condition;
	inertIf?: Condition;
	derivedFrom?: string[];
	default?: JsonValue;
	hint?: string;
	description?: string;
	vocabulary?: string;
	[extra: string]: unknown;
}

export interface Reference {
	modality?: string | null;
	dataset?: { [key: string]: unknown };
	identity: string;
	manifest?: { hash?: string; count?: number };
	view?: { source: string; region: number[] };
	[extra: string]: unknown;
}

export interface Population {
	unit?: string;
	idScheme?: string;
	count?: number;
	manifest?: string;
	accounting?: { [key: string]: unknown };
	[extra: string]: unknown;
}

export interface Stratification {
	label?: string;
	variable?: string | null;
	unit?: string | null;
	/** n+1 ascending edges; the first and/or last may be null (open end: −∞ / +∞). */
	edges?: (number | null)[] | null;
	labels?: string[] | null;
	closed?: string | null;
	lastClosed?: boolean | null;
	total?: string | null;
	categories?: string[] | null;
	after?: string[] | null;
	scheme?: string | null;
	description?: string;
	[extra: string]: unknown;
}

export interface Reducer {
	recompute: RecomputeOp | string;
	over?: string;
	label?: string;
	description?: string;
	vocabulary?: string;
	[extra: string]: unknown;
}

export interface AxisRole {
	role: AxisRoleName | string;
	stratification?: string;
	[extra: string]: unknown;
}

export interface MetricDefinition {
	id?: string;
	kind?: MetricKind | string;
	rev?: number;
	displayName?: string;
	description?: string;
	direction?: Direction | string;
	bounds?: [number | null, number | null];
	unit?: string | null;
	display?: { scale: DisplayScale | string; fullAt?: number; unit?: string | null };
	perItem?: boolean;
	pool?: { form: PoolForm | string; weight?: string };
	/** Reducer (a protocol.reducers name) of cells without a reduction-axis value. */
	reducer?: string;
	requires?: string[];
	dependsOn?: string[];
	vocabulary?: string;
	[extra: string]: unknown;
}

export interface Items {
	ids?: string[];
	series?: { [name: string]: (number | null)[] };
}

/** A metricSet axis declaration (euler-metric-naming `AxisDeclaration`). */
export interface AxisDeclaration {
	position: number;
	values: string[];
	optional?: boolean;
	description?: string;
}

/** The `protocol` block as written by producers (section 4). */
export interface Protocol {
	schemaVersion: string;
	task?: string | null;
	vocabularies?: { [id: string]: string };
	parameters: { [key: string]: JsonValue };
	descriptors?: { [key: string]: Descriptor };
	references?: { [role: string]: Reference };
	observations?: { [key: string]: JsonValue };
	population?: Population;
	stratifications?: { [id: string]: Stratification };
	reducers?: { [value: string]: Reducer };
	axes?: { [axis: string]: AxisRole };
	metrics?: { [base: string]: MetricDefinition };
	items?: Items;
	[extra: string]: unknown;
}

/** A normalized contract: every block present, plus the metricSet namespace and axes. */
export interface Contract {
	schemaVersion: unknown;
	task: string | null;
	vocabularies: { [id: string]: string };
	parameters: { [key: string]: JsonValue };
	descriptors: { [key: string]: Descriptor };
	references: { [role: string]: Reference };
	observations: { [key: string]: JsonValue };
	population: Population;
	stratifications: { [id: string]: Stratification };
	reducers: { [value: string]: Reducer };
	axes: { [axis: string]: AxisRole };
	metrics: { [base: string]: MetricDefinition };
	items: Items;
	metricSet: { metricNamespace: string | null; axes: { [axis: string]: AxisDeclaration } };
	/** Set by the legacy adapter: `<table id>@<version>`. */
	adapter?: string;
	[extra: string]: unknown;
}

export interface Measure {
	kind: ObserveKind;
	d: number;
}

export interface Reason {
	tier: Tier;
	code: ReasonCode;
	key?: string;
	label: string;
	text: string;
	measure?: Measure;
	hint?: string;
}

export interface Verdict {
	tier: Tier;
	label: string;
	/** N of the population rule when it compared id sets (paired, partial, disjoint). */
	shared: number | null;
	reasons: Reason[];
}

export interface CompareOptions {
	/** Evaluated item ids of side A; null/undefined = unknown. */
	itemsA?: Iterable<string> | null;
	itemsB?: Iterable<string> | null;
	/** Per-item names with values on side A (per-file names and items.series); null = none known. */
	perItemA?: Iterable<string> | null;
	perItemB?: Iterable<string> | null;
}

export interface DefinitionKeyObject {
	task: string | null;
	references: { [role: string]: { source: string } | { identity: unknown } };
	parameters: { [key: string]: unknown };
	stratifications: { [id: string]: { [key: string]: unknown } };
	metrics: { [base: string]: number };
}

export interface EvalFile {
	raw: { [key: string]: unknown };
	namespace: string | null;
	axes: { [axis: string]: AxisDeclaration };
	metricSet: { [key: string]: unknown };
	protocol: { [key: string]: unknown } | null;
	contract: Contract;
	/** Aggregate scalar metrics by dotted name. */
	metrics: { [name: string]: number };
	/** Per-file scalar metrics: item id -> dotted name -> value. */
	perFile: { [item: string]: { [name: string]: number } };
	fileIds: string[];
	series: { [name: string]: unknown[] };
	/** items.ids, else the per-file ids, else null (unknown). */
	itemIds: string[] | null;
}

export interface Evidence {
	items: Set<string> | null;
	perItemNames: Set<string>;
}

export type Sha256Hex = (bytes: Uint8Array) => string;

export class CanonicalizationError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'CanonicalizationError';
	}
}

type Rec = { [key: string]: unknown };

/** Sentinel for "not recorded", distinct from JSON null. */
const MISSING: unique symbol = Symbol('missing');

// -- the closed set (section 3) --------------------------------------------------------

export const SCHEMA_VERSION: string = CLOSED_SETS.schemaVersion;
export const SUPPORTED_MAJORS: readonly number[] = CLOSED_SETS.supportedMajors;
const ROLE_DEFAULT_COMPARE: { readonly [role: string]: string } = CLOSED_SETS.roleDefaultCompare;
const COMPARE_RULES: readonly string[] = CLOSED_SETS.compareRules;
const AXIS_ROLES: readonly string[] = CLOSED_SETS.axisRoles;
const CONDITION_OPS: readonly string[] = CLOSED_SETS.conditionOps;
const TIERS: readonly string[] = CLOSED_SETS.tiers;
const ITEM_ID_SEPARATOR: string = CLOSED_SETS.itemIdSeparator;
const STRICTNESS: { [rule: string]: number } = { equal: 3, measure: 2, pair: 1, ignore: 0 };

// -- small helpers -------------------------------------------------------------------------

function isRecord(value: unknown): value is Rec {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function rec(value: unknown): Rec {
	return isRecord(value) ? value : {};
}

function hasOwn(value: Rec, key: string): boolean {
	return Object.prototype.hasOwnProperty.call(value, key);
}

/** `obj.get(key)` of Python: the own value, else null. */
function get(value: Rec, key: string): unknown {
	return hasOwn(value, key) ? value[key] : null;
}

function isNumber(value: unknown): value is number {
	return typeof value === 'number';
}

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function clone<T>(value: T): T {
	if (Array.isArray(value)) return value.map((item) => clone(item)) as T;
	if (isRecord(value)) {
		const out: Rec = {};
		for (const key of Object.keys(value)) out[key] = clone(value[key]);
		return out as T;
	}
	return value;
}

/** Order strings by Unicode code point (Python's default; equals UTF-8 byte order). */
export function compareCodePoints(a: string, b: string): number {
	let i = 0;
	let j = 0;
	while (i < a.length && j < b.length) {
		const x = a.codePointAt(i) as number;
		const y = b.codePointAt(j) as number;
		if (x !== y) return x < y ? -1 : 1;
		i += x > 0xffff ? 2 : 1;
		j += y > 0xffff ? 2 : 1;
	}
	const restA = a.length - i;
	const restB = b.length - j;
	return restA === restB ? 0 : restA < restB ? -1 : 1;
}

function sortedCodePoints(values: Iterable<string>): string[] {
	return [...values].sort(compareCodePoints);
}

/** JSON equality of section 8.0. */
export function jsonEqual(a: unknown, b: unknown): boolean {
	if (typeof a === 'boolean' || typeof b === 'boolean') {
		return typeof a === 'boolean' && typeof b === 'boolean' && a === b;
	}
	if (isNumber(a) && isNumber(b)) return a === b;
	if (typeof a === 'string' && typeof b === 'string') return a === b;
	if (a === null || a === undefined || b === null || b === undefined) {
		return (a === null || a === undefined) && (b === null || b === undefined);
	}
	if (Array.isArray(a) && Array.isArray(b)) {
		return a.length === b.length && a.every((item, index) => jsonEqual(item, b[index]));
	}
	if (isRecord(a) && isRecord(b)) {
		const keysA = Object.keys(a);
		const keysB = Object.keys(b);
		if (keysA.length !== keysB.length) return false;
		const setB = new Set(keysB);
		if (!keysA.every((key) => setB.has(key))) return false;
		return keysA.every((key) => jsonEqual(a[key], b[key]));
	}
	return false;
}

// -- JCS (section 9.1) ----------------------------------------------------------------------

const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
const ESCAPES: { [char: string]: string } = {
	'"': '\\"',
	'\\': '\\\\',
	'\b': '\\b',
	'\f': '\\f',
	'\n': '\\n',
	'\r': '\\r',
	'\t': '\\t'
};

/** ECMAScript Number.prototype.toString, as JCS requires (-0 is "0"). */
export function formatNumber(value: number | bigint): string {
	if (typeof value === 'bigint') {
		if (value > MAX_SAFE || value < -MAX_SAFE) {
			throw new CanonicalizationError(`integer ${value} is outside the safe range ±(2**53 - 1)`);
		}
		return value.toString();
	}
	if (!Number.isFinite(value)) throw new CanonicalizationError(`non-finite number ${value}`);
	if (value === 0) return '0';
	return String(value);
}

/** The JSON string literal JSON.stringify writes; lone surrogates are errors. */
export function quote(text: string): string {
	let out = '"';
	for (let i = 0; i < text.length; i++) {
		const ch = text[i] as string;
		const escape = ESCAPES[ch];
		if (escape !== undefined) {
			out += escape;
			continue;
		}
		const code = text.charCodeAt(i);
		if (code < 0x20) {
			out += '\\u' + code.toString(16).padStart(4, '0');
		} else if (code >= 0xd800 && code <= 0xdbff) {
			const next = i + 1 < text.length ? text.charCodeAt(i + 1) : 0;
			if (next < 0xdc00 || next > 0xdfff) throw new CanonicalizationError('lone surrogate in string');
			out += ch + text[i + 1];
			i += 1;
		} else if (code >= 0xdc00 && code <= 0xdfff) {
			throw new CanonicalizationError('lone surrogate in string');
		} else {
			out += ch;
		}
	}
	return out + '"';
}

/** RFC 8785 canonical text of a JSON value (bigint for integer-typed numbers). */
export function canonicalize(value: unknown): string {
	if (value === null) return 'null';
	if (value === true) return 'true';
	if (value === false) return 'false';
	if (typeof value === 'number' || typeof value === 'bigint') return formatNumber(value);
	if (typeof value === 'string') return quote(value);
	if (Array.isArray(value)) return '[' + value.map((item) => canonicalize(item)).join(',') + ']';
	if (isRecord(value)) {
		const keys = Object.keys(value);
		const quoted: { [key: string]: string } = {};
		for (const key of keys) quoted[key] = quote(key);
		keys.sort((x, y) => (x < y ? -1 : x > y ? 1 : 0)); // UTF-16 code units
		return '{' + keys.map((key) => quoted[key] + ':' + canonicalize(value[key])).join(',') + '}';
	}
	throw new CanonicalizationError(`not a JSON value: ${typeof value}`);
}

// -- SHA-256 and hashes (sections 9.2 - 9.4) ---------------------------------------------------

const K256 = new Uint32Array([
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]);

function rotr(x: number, n: number): number {
	return (x >>> n) | (x << (32 - n));
}

/** Pure-TS SHA-256 (lower-case hex). Swap in node:crypto via euler-contract-node.ts if preferred. */
export function sha256Hex(bytes: Uint8Array): string {
	const h = new Uint32Array([
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	]);
	const length = bytes.length;
	const total = Math.ceil((length + 9) / 64) * 64;
	const buffer = new Uint8Array(total);
	buffer.set(bytes);
	buffer[length] = 0x80;
	const view = new DataView(buffer.buffer);
	view.setUint32(total - 8, Math.floor(length / 0x20000000));
	view.setUint32(total - 4, (length << 3) >>> 0);
	const w = new Uint32Array(64);
	for (let offset = 0; offset < total; offset += 64) {
		for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4);
		for (let i = 16; i < 64; i++) {
			const w15 = w[i - 15] as number;
			const w2 = w[i - 2] as number;
			const s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >>> 3);
			const s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >>> 10);
			w[i] = ((w[i - 16] as number) + s0 + (w[i - 7] as number) + s1) >>> 0;
		}
		let a = h[0] as number;
		let b = h[1] as number;
		let c = h[2] as number;
		let d = h[3] as number;
		let e = h[4] as number;
		let f = h[5] as number;
		let g = h[6] as number;
		let hh = h[7] as number;
		for (let i = 0; i < 64; i++) {
			const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
			const ch = (e & f) ^ (~e & g);
			const t1 = (hh + s1 + ch + (K256[i] as number) + (w[i] as number)) >>> 0;
			const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
			const maj = (a & b) ^ (a & c) ^ (b & c);
			const t2 = (s0 + maj) >>> 0;
			hh = g;
			g = f;
			f = e;
			e = (d + t1) >>> 0;
			d = c;
			c = b;
			b = a;
			a = (t1 + t2) >>> 0;
		}
		h[0] = ((h[0] as number) + a) >>> 0;
		h[1] = ((h[1] as number) + b) >>> 0;
		h[2] = ((h[2] as number) + c) >>> 0;
		h[3] = ((h[3] as number) + d) >>> 0;
		h[4] = ((h[4] as number) + e) >>> 0;
		h[5] = ((h[5] as number) + f) >>> 0;
		h[6] = ((h[6] as number) + g) >>> 0;
		h[7] = ((h[7] as number) + hh) >>> 0;
	}
	let hex = '';
	for (const word of h) hex += word.toString(16).padStart(8, '0');
	return hex;
}

/** UTF-8 bytes of a string (lone surrogates become U+FFFD; canonical text has none). */
export function utf8Encode(text: string): Uint8Array {
	const out: number[] = [];
	for (let i = 0; i < text.length; i++) {
		let code = text.charCodeAt(i);
		if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
			const next = text.charCodeAt(i + 1);
			if (next >= 0xdc00 && next <= 0xdfff) {
				code = 0x10000 + ((code - 0xd800) << 10) + (next - 0xdc00);
				i += 1;
			}
		}
		if (code >= 0xd800 && code <= 0xdfff) code = 0xfffd;
		if (code < 0x80) out.push(code);
		else if (code < 0x800) out.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
		else if (code < 0x10000) out.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
		else
			out.push(
				0xf0 | (code >> 18),
				0x80 | ((code >> 12) & 0x3f),
				0x80 | ((code >> 6) & 0x3f),
				0x80 | (code & 0x3f)
			);
	}
	return new Uint8Array(out);
}

/** `"sha256:" + hex(SHA-256(UTF-8(JCS(value))))` (section 9.2). */
export function hashValue(value: unknown, sha: Sha256Hex = sha256Hex): string {
	return 'sha256:' + sha(utf8Encode(canonicalize(value)));
}

/** Item ids NFC-normalized, de-duplicated, sorted by code point (section 9.3). */
export function manifestIds(ids: Iterable<string>): string[] {
	const normalized = new Set<string>();
	for (const item of ids) {
		if (typeof item !== 'string') throw new TypeError(`item id ${String(item)} is not a string`);
		normalized.add(item.normalize('NFC'));
	}
	return sortedCodePoints(normalized);
}

/** The population manifest hash (section 9.3). */
export function manifestHash(ids: Iterable<string>, sha: Sha256Hex = sha256Hex): string {
	return hashValue(manifestIds(ids), sha);
}

/** A recorded `rev`: an integral number >= 1. */
export function validRev(rev: unknown): rev is number {
	return isNumber(rev) && Number.isInteger(rev) && rev >= 1;
}

/** The object whose hash is the definition key (section 9.4). */
export function definitionKeyObject(contract: Contract | Rec): DefinitionKeyObject {
	const source = contract as Rec;
	const parameters = rec(source.parameters);
	const observations = rec(source.observations);
	const descriptors = rec(source.descriptors);
	const task = source.task;

	const references: DefinitionKeyObject['references'] = {};
	for (const [role, reference] of Object.entries(rec(source.references))) {
		if (!isRecord(reference)) continue;
		const view = reference.view;
		if (isRecord(view) && typeof view.source === 'string') references[role] = { source: view.source };
		else references[role] = { identity: get(reference, 'identity') };
	}

	const params: { [key: string]: unknown } = {};
	for (const [key, value] of Object.entries(parameters)) {
		const raw = descriptors[key];
		const descriptor = isRecord(raw) ? raw : null;
		if (effectiveCompare(descriptor) !== 'equal') continue;
		const applies = descriptor ? get(descriptor, 'appliesWhen') : null;
		if (applies !== null && evaluateCondition(applies, parameters, observations) === false) continue;
		params[key] = value;
	}

	const stratifications: DefinitionKeyObject['stratifications'] = {};
	for (const [id, stratification] of Object.entries(rec(source.stratifications))) {
		if (isRecord(stratification)) stratifications[id] = identityProjection(stratification);
	}

	const metrics: { [base: string]: number } = {};
	for (const [base, definition] of Object.entries(rec(source.metrics))) {
		const rev = isRecord(definition) ? definition.rev : null;
		if (validRev(rev)) metrics[base] = rev;
	}

	return {
		task: typeof task === 'string' ? task : null,
		references,
		parameters: params,
		stratifications,
		metrics
	};
}

/** The definition key of a normalized contract (section 9.4). */
export function definitionKey(contract: Contract | Rec, sha: Sha256Hex = sha256Hex): string {
	return hashValue(definitionKeyObject(contract), sha);
}

// -- conditions (section 7) ------------------------------------------------------------------

const FORMS = ['all', 'any', 'not', 'param', 'observation'];

/** Evaluate a condition on one side: true, false or null (unknown). */
export function evaluateCondition(condition: unknown, parameters?: unknown, observations?: unknown): Truth {
	return evaluate(condition, rec(parameters), rec(observations));
}

function evaluate(condition: unknown, params: Rec, obs: Rec): Truth {
	if (!isRecord(condition)) return null;
	const forms = FORMS.filter((form) => hasOwn(condition, form));
	if (forms.length !== 1) return null;
	const form = forms[0] as string;
	if (form === 'all' || form === 'any') {
		const members = condition[form];
		if (!Array.isArray(members)) return null;
		const results = members.map((member) => evaluate(member, params, obs));
		if (form === 'all') {
			if (results.some((result) => result === false)) return false;
			if (results.some((result) => result === null)) return null;
			return true;
		}
		if (results.some((result) => result === true)) return true;
		if (results.some((result) => result === null)) return null;
		return false;
	}
	if (form === 'not') {
		const result = evaluate(condition.not, params, obs);
		return result === null ? null : !result;
	}
	const key = condition[form];
	const op = get(condition, 'op');
	if (typeof key !== 'string' || typeof op !== 'string' || !CONDITION_OPS.includes(op)) return null;
	const source = form === 'param' ? params : obs;
	const present = hasOwn(source, key) && source[key] !== null && source[key] !== undefined;
	if (op === 'exists') return present;
	if (!present || !hasOwn(condition, 'value')) return null;
	const operand = source[key];
	const value = condition.value;
	if (op === 'eq') return jsonEqual(operand, value);
	if (op === 'ne') return !jsonEqual(operand, value);
	if (op === 'in') {
		if (!Array.isArray(value)) return null;
		return value.some((item) => jsonEqual(operand, item));
	}
	if (!(isNumber(operand) && isNumber(value))) return null;
	if (op === 'lt') return operand < value;
	if (op === 'le') return operand <= value;
	if (op === 'gt') return operand > value;
	return operand >= value;
}

// -- globs (section 4.2) -------------------------------------------------------------------

/** `*` matches any run (dots included); everything else literal; anchored, case-sensitive. */
export function globMatch(pattern: string, name: string): boolean {
	const parts = pattern.split('*');
	if (parts.length === 1) return pattern === name;
	const first = parts[0] as string;
	const last = parts[parts.length - 1] as string;
	if (name.length < first.length + last.length) return false;
	if (!name.startsWith(first) || !name.endsWith(last)) return false;
	let position = first.length;
	const end = name.length - last.length;
	for (const middle of parts.slice(1, -1)) {
		if (!middle) continue;
		const index = name.indexOf(middle, position);
		if (index < 0 || index + middle.length > end) return false;
		position = index + middle.length;
	}
	return true;
}

function globAny(patterns: unknown[], name: string): boolean {
	return patterns.some((pattern) => typeof pattern === 'string' && globMatch(pattern, name));
}

// -- measures and phrases (section 3.3) --------------------------------------------------------

/** d >= 0 for two observed values, or null when unmeasurable. */
export function measure(kind: unknown, a: unknown, b: unknown): number | null {
	if (kind === 'region') return regionMeasure(a, b);
	if (kind === 'count') return countMeasure(a, b);
	if (kind === 'fraction') return fractionMeasure(a, b);
	if (kind === 'factor') return factorMeasure(a, b);
	if (kind === 'size') return sizeMeasure(a, b);
	return null;
}

function nonneg(value: unknown): number | null {
	if (!isNumber(value) || !Number.isFinite(value) || value < 0) return null;
	return value;
}

function regionValue(value: unknown): [string, number, number, number, number] | null {
	if (!isRecord(value) || typeof value.frame !== 'string') return null;
	const box = value.box;
	if (!Array.isArray(box) || box.length !== 4) return null;
	const numbers = box.map((item) => nonneg(item));
	if (numbers.some((number) => number === null)) return null;
	const [top, left, height, width] = numbers as number[];
	return [value.frame, top as number, left as number, height as number, width as number];
}

function regionMeasure(a: unknown, b: unknown): number | null {
	const ra = regionValue(a);
	const rb = regionValue(b);
	if (ra === null || rb === null || ra[0] !== rb[0]) return null;
	const [, top1, left1, height1, width1] = ra;
	const [, top2, left2, height2, width2] = rb;
	const interH = Math.max(0, Math.min(top1 + height1, top2 + height2) - Math.max(top1, top2));
	const interW = Math.max(0, Math.min(left1 + width1, left2 + width2) - Math.max(left1, left2));
	const inter = interH * interW;
	const union = height1 * width1 + height2 * width2 - inter;
	if (!(union > 0)) return null;
	return 1 - inter / union;
}

function relative(a: number, b: number): number {
	if (a === 0 && b === 0) return 0;
	return Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b));
}

function countMeasure(a: unknown, b: unknown): number | null {
	const na = nonneg(a);
	const nb = nonneg(b);
	if (na === null || nb === null) return null;
	return relative(na, nb);
}

function fractionMeasure(a: unknown, b: unknown): number | null {
	const na = nonneg(a);
	const nb = nonneg(b);
	if (na === null || nb === null || na > 1 || nb > 1) return null;
	return Math.abs(na - nb);
}

function factorValues(value: unknown): number[] | null {
	const items = Array.isArray(value) ? value : [value];
	if (items.length === 0) return null;
	const numbers = items.map((item) => nonneg(item));
	if (numbers.some((number) => number === null || number <= 0)) return null;
	return numbers as number[];
}

function factorMeasure(a: unknown, b: unknown): number | null {
	if (Array.isArray(a) !== Array.isArray(b)) return null;
	const fa = factorValues(a);
	const fb = factorValues(b);
	if (fa === null || fb === null || fa.length !== fb.length) return null;
	let worst = 1;
	fa.forEach((x, index) => {
		const y = fb[index] as number;
		worst = Math.max(worst, x / y, y / x);
	});
	return worst - 1;
}

function sizeProduct(value: unknown): number | null {
	if (!Array.isArray(value) || value.length === 0) return null;
	let product = 1;
	for (const item of value) {
		const number = nonneg(item);
		if (number === null) return null;
		product = product * number;
	}
	return product;
}

function sizeMeasure(a: unknown, b: unknown): number | null {
	if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return null;
	const pa = sizeProduct(a);
	const pb = sizeProduct(b);
	if (pa === null || pb === null) return null;
	return relative(pa, pb);
}

/** d in percent with one decimal: floor(d*1000 + 0.5) / 10; "<0.1" for tiny non-zero d. */
export function formatPercent(d: number): string {
	const t = Math.floor(d * 1000 + 0.5);
	if (t === 0 && d !== 0) return '<0.1';
	return `${Math.floor(t / 10)}.${t % 10}`;
}

function stripFraction(text: string): string {
	if (!text.includes('.')) return text;
	return text.replace(/0+$/, '').replace(/\.$/, '');
}

/** toPrecision(3) with trailing fraction zeros removed. */
export function formatSig3(value: number): string {
	if (!Number.isFinite(value)) return String(value);
	if (value === 0) return '0';
	const text = value.toPrecision(3);
	const at = text.indexOf('e');
	if (at >= 0) return stripFraction(text.slice(0, at)) + text.slice(at);
	return stripFraction(text);
}

/** Factor value(s) for phrases: 3 significant digits, lists joined by ×. */
export function formatFactor(value: unknown): string {
	const items = Array.isArray(value) ? value : [value];
	return items.map((item) => (isNumber(item) ? formatSig3(item) : '?')).join('×');
}

/** The verdict phrase of a measured difference (section 3.3). */
export function phrase(kind: string, label: string, a: unknown, b: unknown, d: number): string {
	if (kind === 'factor') return `${label} (${formatFactor(a)} vs ${formatFactor(b)})`;
	const unit = kind === 'fraction' ? 'pp' : '%';
	return `${label} differs by ${formatPercent(d)} ${unit}`;
}

// -- stratifications (section 4.6) ---------------------------------------------------------------

const DISPLAY_FIELDS = new Set(['label', 'description', 'unit']);

/** A stratification without display fields and null-valued fields. */
export function identityProjection(stratification: unknown): { [key: string]: unknown } {
	const out: { [key: string]: unknown } = {};
	if (!isRecord(stratification)) return out;
	for (const [key, value] of Object.entries(stratification)) {
		if (!DISPLAY_FIELDS.has(key) && value !== null && value !== undefined) out[key] = value;
	}
	return out;
}

/** Edges of section 4.6: at least two; inner ones numbers; the first and last may be null. */
export function validEdges(edges: unknown): edges is (number | null)[] {
	if (!Array.isArray(edges) || edges.length < 2) return false;
	const last = edges.length - 1;
	return edges.every((edge, index) => isNumber(edge) || (edge === null && (index === 0 || index === last)));
}

/** [lo, hi] of a bin label (null = open end); the total spans first to last edge. */
function binRange(stratification: unknown, label: string): [number | null, number | null] | null {
	if (!isRecord(stratification)) return null;
	const edges = stratification.edges;
	if (!validEdges(edges)) return null;
	if (label === stratification.total) return [edges[0] as number | null, edges[edges.length - 1] as number | null];
	const labels = stratification.labels;
	if (!Array.isArray(labels) || labels.length !== edges.length - 1) return null;
	const index = labels.indexOf(label);
	if (index < 0) return null;
	return [edges[index] as number | null, edges[index + 1] as number | null];
}

/** Edges of the sqrt_equal_width scheme (euler-eval's benchmark bins). */
export function sqrtEqualWidthEdges(lo: number, hi: number, bins = 3): number[] {
	if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < 0 || hi <= lo) {
		throw new RangeError(`invalid range [${lo}, ${hi}]`);
	}
	if (bins < 1) throw new RangeError('bins must be >= 1');
	const rootLo = Math.sqrt(lo);
	const rootHi = Math.sqrt(hi);
	const step = (rootHi - rootLo) / bins;
	const edges = [lo];
	for (let index = 1; index < bins; index++) {
		const root = rootLo + index * step;
		edges.push(root * root);
	}
	edges.push(hi);
	return edges;
}

// -- descriptors (sections 3.1, 4.2, 8.4) -----------------------------------------------------

/** The compare rule a descriptor stands for; unknown roles and values are equal. */
export function effectiveCompare(descriptor: unknown): CompareRule {
	if (!isRecord(descriptor)) return 'equal';
	const role = descriptor.role;
	if (typeof role !== 'string' || !hasOwn(ROLE_DEFAULT_COMPARE, role)) return 'equal';
	if (!hasOwn(descriptor, 'compare')) return ROLE_DEFAULT_COMPARE[role] as CompareRule;
	const compare = descriptor.compare;
	return typeof compare === 'string' && COMPARE_RULES.includes(compare) ? (compare as CompareRule) : 'equal';
}

interface Cell {
	name: string;
	base: string;
	roles: { [role: string]: string[] };
}

function cellValues(cell: Cell, role: string): string[] {
	return hasOwn(cell.roles, role) ? [...(cell.roles[role] as string[])] : [];
}

function cellValue(cell: Cell, role: string): string | null {
	const values = cellValues(cell, role);
	return values.length > 0 ? (values[0] as string) : null;
}

function affectsCell(descriptor: Rec, cell: Cell): boolean {
	const affects = descriptor.affects;
	if (!isRecord(affects)) return true;
	const metrics = hasOwn(affects, 'metrics') ? affects.metrics : ['*'];
	if (Array.isArray(metrics) && !globAny(metrics, cell.base)) return false;
	const axes = affects.axes;
	if (isRecord(axes)) {
		for (const [role, allowed] of Object.entries(axes)) {
			if (!Array.isArray(allowed)) continue;
			if (!cellValues(cell, role).some((value) => allowed.includes(value))) return false;
		}
	}
	return true;
}

type Context = [Rec, Rec];

interface MergedDescriptor {
	key: string;
	compare: CompareRule;
	label: string;
	observe: unknown;
	hasDefault: boolean;
	defaultValue: unknown;
	hint: string | null;
	sides: Rec[];
	candidates: Rec[];
}

function skippedBy(descriptor: Rec, contextA: Context, contextB: Context): boolean {
	const applies = get(descriptor, 'appliesWhen');
	if (applies !== null) {
		if (evaluate(applies, ...contextA) === false && evaluate(applies, ...contextB) === false) return true;
	}
	const inert = get(descriptor, 'inertIf');
	if (inert !== null) {
		if (evaluate(inert, ...contextA) === true && evaluate(inert, ...contextB) === true) return true;
	}
	return false;
}

function mergedAffects(merged: MergedDescriptor, cell: Cell): boolean {
	if (merged.sides.length === 0) return true;
	return merged.sides.some((descriptor) => affectsCell(descriptor, cell));
}

function mergedSkipped(merged: MergedDescriptor, contextA: Context, contextB: Context): boolean {
	if (merged.candidates.length === 0) return false;
	return merged.candidates.every((descriptor) => skippedBy(descriptor, contextA, contextB));
}

function mergeDescriptors(key: string, a: unknown, b: unknown): MergedDescriptor {
	const da = isRecord(a) ? a : null;
	const db = isRecord(b) ? b : null;
	const sides = [da, db].filter((d): d is Rec => d !== null);
	if (sides.length === 0) {
		return {
			key,
			compare: 'equal',
			label: key,
			observe: null,
			hasDefault: false,
			defaultValue: null,
			hint: null,
			sides,
			candidates: []
		};
	}
	let candidates = sides;
	if (da !== null && db !== null) {
		const ca = STRICTNESS[effectiveCompare(da)] as number;
		const cb = STRICTNESS[effectiveCompare(db)] as number;
		if (ca > cb) candidates = [da];
		else if (cb > ca) candidates = [db];
	}
	const first = candidates[0] as Rec;
	const compare = effectiveCompare(first);
	let observe = get(first, 'observe');
	if (candidates.slice(1).some((d) => !jsonEqual(get(d, 'observe'), observe))) observe = null;

	let hasDefault = false;
	let defaultValue: unknown = null;
	const defaults = candidates.filter((d) => hasOwn(d, 'default')).map((d) => d.default);
	if (defaults.length > 0 && defaults.slice(1).every((value) => jsonEqual(value, defaults[0]))) {
		hasDefault = true;
		defaultValue = defaults[0];
	}
	const labelled = candidates.find((d) => typeof d.label === 'string' && d.label !== '');
	const hinted = candidates.find((d) => typeof d.hint === 'string' && d.hint !== '');
	return {
		key,
		compare,
		label: labelled ? (labelled.label as string) : key,
		observe,
		hasDefault,
		defaultValue,
		hint: hinted ? (hinted.hint as string) : null,
		sides,
		candidates
	};
}

// -- metric names: decompose (euler-metric-naming axes.decompose) -------------------------------

export interface DecomposedMetric {
	namespace: string;
	axes: { [axis: string]: string | null };
	metric: string;
}

function sortedDeclarations(axes: { [axis: string]: AxisDeclaration }): [string, AxisDeclaration][] {
	return Object.entries(axes).sort((x, y) => x[1].position - y[1].position);
}

/** Decompose a metric name into namespace, axis values and base metric; throws when it does not. */
export function decompose(
	metricName: string,
	namespace: string,
	axes: { [axis: string]: AxisDeclaration }
): DecomposedMetric {
	const prefix = namespace + '.';
	if (!metricName.startsWith(prefix)) {
		throw new Error(`metric ${metricName} does not belong to namespace ${namespace}`);
	}
	const segments = metricName.slice(prefix.length).split('.');
	if (segments.length === 1 && segments[0] === '') {
		throw new Error(`metric ${metricName} has no segments after namespace`);
	}
	const values: { [axis: string]: string | null } = {};
	let index = 0;
	for (const [axisName, declaration] of sortedDeclarations(axes)) {
		if (index < segments.length) {
			const candidate = segments[index] as string;
			if (declaration.values.includes(candidate) && segments.length - index - 1 >= 1) {
				values[axisName] = candidate;
				index += 1;
				continue;
			}
		}
		if (declaration.optional === true) values[axisName] = null;
		else throw new Error(`required axis ${axisName} missing in metric ${metricName}`);
	}
	const remaining = segments.slice(index);
	if (remaining.length === 0) throw new Error(`metric ${metricName} has no base metric name`);
	return { namespace, axes: values, metric: remaining.join('.') };
}

/** Parse metricSet axis declarations from untrusted JSON; null when malformed. */
export function axisDeclarations(raw: unknown): { [axis: string]: AxisDeclaration } | null {
	if (!isRecord(raw)) return null;
	const out: { [axis: string]: AxisDeclaration } = {};
	for (const [name, declaration] of Object.entries(raw)) {
		if (!isRecord(declaration)) return null;
		const position = declaration.position;
		const values = declaration.values;
		if (!(isNumber(position) && Number.isInteger(position))) return null;
		if (!isStringArray(values)) return null;
		const parsed: AxisDeclaration = { position, values: [...values], optional: declaration.optional === true };
		if (typeof declaration.description === 'string') parsed.description = declaration.description;
		out[name] = parsed;
	}
	return out;
}

// -- normalization (sections 6, 10) -------------------------------------------------------------

const VERSION_RE = /^(\d+)(?:\.(\d+))?$/;
const RECORD_BLOCKS = [
	'vocabularies',
	'parameters',
	'descriptors',
	'references',
	'observations',
	'population',
	'stratifications',
	'reducers',
	'axes',
	'metrics',
	'items'
];

/** The major of a "MAJOR.MINOR" (or "MAJOR") schema version, else null. */
export function schemaMajor(version: unknown): number | null {
	if (typeof version !== 'string') return null;
	const match = VERSION_RE.exec(version.trim());
	return match ? Number(match[1]) : null;
}

/** Whether the engine can judge this contract beyond section 8.8. */
export function isSupported(contract: Contract | Rec): boolean {
	const major = schemaMajor((contract as Rec).schemaVersion);
	return major !== null && SUPPORTED_MAJORS.includes(major);
}

/** `{metricNamespace, axes}` of a metricSet envelope (camel or snake case). */
export function metricSetInfo(metricSet: unknown): Contract['metricSet'] {
	if (!isRecord(metricSet)) return { metricNamespace: null, axes: {} };
	let namespace: string | null = null;
	for (const key of ['metricNamespace', 'metric_namespace', 'namespace']) {
		const value = metricSet[key];
		if (typeof value === 'string' && value.trim()) {
			namespace = value.trim();
			break;
		}
	}
	const axes = metricSet.axes;
	return { metricNamespace: namespace, axes: isRecord(axes) ? (clone(axes) as Contract['metricSet']['axes']) : {} };
}

function emptyContract(schemaVersion: unknown): Contract {
	return {
		schemaVersion,
		task: null,
		vocabularies: {},
		parameters: {},
		descriptors: {},
		references: {},
		observations: {},
		population: {},
		stratifications: {},
		reducers: {},
		axes: {},
		metrics: {},
		items: {},
		metricSet: { metricNamespace: null, axes: {} }
	};
}

/** Normalize a results file's protocol block (any schema version) into a contract. */
export function normalize(protocol: unknown, metricSet?: unknown, meta?: unknown): Contract {
	const source = rec(protocol);
	if (schemaMajor(source.schemaVersion) === 1) return normalizeLegacy(source, metricSet, meta);
	const contract = emptyContract(source.schemaVersion) as unknown as Rec;
	for (const [key, value] of Object.entries(source)) contract[key] = clone(value);
	if (typeof contract.task !== 'string') contract.task = null;
	for (const block of RECORD_BLOCKS) {
		if (!isRecord(contract[block])) contract[block] = {};
	}
	contract.metricSet = metricSetInfo(metricSet);
	return contract as unknown as Contract;
}

/** The frozen schema-1 adapter table (contract/legacy/protocol-1.json). */
export interface LegacyTable {
	id: string;
	version: string;
	references: { prefix: string; fields: string[]; modalityField: string; roleRenames: { [role: string]: string } };
	population: { count: string };
	parameters: { [v1: string]: string };
	observations: { [v1: string]: string };
	applied: { [v1: string]: string };
	appliedFrom: string;
	benchmark: {
		key: string;
		parameter: string;
		stratification: string;
		template: { [key: string]: unknown; labels: string[] };
		boundariesFrom: string;
	};
	revision: { key: string; lastKnown: number; beyondKey: string; policies: { [key: string]: [number, string][] } };
	resample: {
		key: string;
		prediction: { methods: string[]; observation: string };
		gt: { methods: string[]; parameter: string };
	};
	axisRoles: { [axis: string]: AxisRole };
	reducers: { [value: string]: Reducer };
	descriptors: { [key: string]: Descriptor };
}

const TABLE = LEGACY_PROTOCOL_1 as unknown as LegacyTable;

/** The frozen v1 adapter table (a deep copy). */
export function legacyTable(): LegacyTable {
	return clone(TABLE);
}

function dims(meta: unknown, side: string): [number, number] | null {
	const block = isRecord(meta) ? meta[side] : null;
	const dimensions = isRecord(block) ? block.dimensions : null;
	if (!isRecord(dimensions)) return null;
	const height = dimensions.height;
	const width = dimensions.width;
	if (!(isNumber(height) && isNumber(width) && height > 0 && width > 0)) return null;
	return [height, width];
}

function ratio(target: [number, number] | null, source: [number, number] | null): number[] | null {
	if (target === null || source === null) return null;
	return [target[0] / source[0], target[1] / source[1]];
}

function benchmarkEdges(value: unknown, labels: string[], boundaries: unknown): number[] | null {
	if (
		!(
			Array.isArray(value) &&
			value.length === 2 &&
			value.every(isNumber) &&
			0 <= (value[0] as number) &&
			(value[0] as number) < (value[1] as number)
		)
	) {
		return null;
	}
	if (isRecord(boundaries) && jsonEqual(get(boundaries, 'range'), value)) {
		const bins = labels.map((label) => get(boundaries, label));
		if (bins.every((bin) => Array.isArray(bin) && bin.length === 2 && bin.every(isNumber))) {
			const lists = bins as number[][];
			return [(lists[0] as number[])[0] as number, ...lists.map((bin) => bin[1] as number)];
		}
	}
	return sqrtEqualWidthEdges(value[0] as number, value[1] as number, labels.length);
}

/** Normalize a schema-1.x protocol block into a v2 contract (section 10). */
export function normalizeLegacy(protocol: unknown, metricSet?: unknown, meta?: unknown): Contract {
	const table = TABLE;
	const source = rec(protocol);
	const v1 = rec(source.parameters);
	const contract = emptyContract(source.schemaVersion);
	contract.adapter = `${table.id}@${table.version}`;
	contract.metricSet = metricSetInfo(metricSet);
	const metadata = rec(isRecord(metricSet) ? metricSet.metadata : null);
	const descriptors = table.descriptors;
	const parameters = contract.parameters as Rec;
	const observations = contract.observations as Rec;
	const consumed = new Set<string>();

	const describe = (key: string): void => {
		if (hasOwn(descriptors, key)) contract.descriptors[key] = clone(descriptors[key] as Descriptor);
	};
	const setParam = (key: string, value: unknown): void => {
		parameters[key] = clone(value);
		describe(key);
	};

	// references: reference.<role>.<field>
	const spec = table.references;
	const fields = spec.fields;
	const accepted = new Set([...fields, spec.modalityField]);
	const roles: { [role: string]: Rec } = {};
	for (const [key, value] of Object.entries(v1)) {
		if (!key.startsWith(spec.prefix)) continue;
		const rest = key.slice(spec.prefix.length);
		const dot = rest.indexOf('.');
		if (dot < 0) continue;
		const role = rest.slice(0, dot);
		const field = rest.slice(dot + 1);
		if (!role || !accepted.has(field)) continue;
		(roles[role] ??= {})[field] = value;
		consumed.add(key);
	}
	for (const [role, values] of Object.entries(roles)) {
		if (fields.every((field) => get(values, field) === null)) continue;
		const name = hasOwn(spec.roleRenames, role) ? (spec.roleRenames[role] as string) : role;
		const modality = get(values, spec.modalityField);
		const identitySource: Rec = { [spec.modalityField]: modality };
		for (const field of fields) identitySource[field] = get(values, field);
		const reference: Rec = {};
		if (modality !== null) reference.modality = modality;
		reference.dataset = { id: get(values, 'dataset'), scope: get(values, 'scope'), split: get(values, 'split') };
		reference.identity = hashValue(identitySource);
		contract.references[name] = reference as unknown as Reference;
		describe(`reference.${name}`);
	}

	// population
	const countKey = table.population.count;
	if (hasOwn(v1, countKey)) {
		consumed.add(countKey);
		if (isNumber(v1[countKey])) contract.population.count = v1[countKey] as number;
	}

	// plain renames
	for (const [v1Key, key] of Object.entries(table.parameters)) {
		if (hasOwn(v1, v1Key)) {
			consumed.add(v1Key);
			setParam(key, v1[v1Key]);
		}
	}
	for (const [v1Key, key] of Object.entries(table.observations)) {
		if (hasOwn(v1, v1Key)) {
			consumed.add(v1Key);
			observations[key] = clone(v1[v1Key]);
			describe(key);
		}
	}

	// whether an alignment was applied, from metricSet.metadata
	const applied = hasOwn(metadata, table.appliedFrom) ? metadata[table.appliedFrom] : undefined;
	for (const [v1Key, key] of Object.entries(table.applied)) {
		if (hasOwn(v1, v1Key) && typeof applied === 'boolean') setParam(key, applied);
	}

	// benchmark range: a derived parameter and the GT-depth stratification
	const bench = table.benchmark;
	if (hasOwn(v1, bench.key)) {
		consumed.add(bench.key);
		const value = v1[bench.key];
		setParam(bench.parameter, value);
		const template = bench.template;
		let boundaries: unknown = metadata;
		for (const part of bench.boundariesFrom.split('.')) {
			boundaries = isRecord(boundaries) ? get(boundaries, part) : null;
		}
		const edges = benchmarkEdges(value, template.labels, boundaries);
		if (edges !== null) {
			const stratification: Rec = {};
			for (const [key, item] of Object.entries(template)) {
				stratification[key] = clone(item);
				if (key === 'unit') stratification.edges = edges;
			}
			if (!hasOwn(stratification, 'edges')) stratification.edges = edges;
			contract.stratifications[bench.stratification] = stratification as Stratification;
		}
	}

	// metrics revision: the policies it implies
	const revision = table.revision;
	if (hasOwn(v1, revision.key)) {
		consumed.add(revision.key);
		const r = v1[revision.key];
		if (isNumber(r) && Number.isInteger(r) && r >= 1) {
			for (const [key, steps] of Object.entries(revision.policies)) {
				let chosen = (steps[0] as [number, string])[1];
				for (const [start, value] of steps) {
					if (r >= start) chosen = value;
				}
				setParam(key, chosen);
			}
			if (r > revision.lastKnown) setParam(revision.beyondKey, r);
		} else {
			setParam(revision.beyondKey, r);
		}
	}

	// resampling factors from meta dimensions
	const resample = table.resample;
	const method = v1[resample.key];
	const gtDims = dims(meta, 'gt');
	const predDims = dims(meta, 'pred');
	if (typeof method === 'string' && resample.prediction.methods.includes(method)) {
		const key = resample.prediction.observation;
		observations[key] = ratio(gtDims, predDims);
		describe(key);
	} else if (typeof method === 'string' && resample.gt.methods.includes(method)) {
		setParam(resample.gt.parameter, ratio(predDims, gtDims));
	}

	// everything else passes through, undescribed (compared strictly)
	for (const [key, value] of Object.entries(v1)) {
		if (!consumed.has(key)) parameters[key] = clone(value);
	}

	// axis roles and reducers
	for (const name of Object.keys(contract.metricSet.axes)) {
		if (hasOwn(table.axisRoles, name)) contract.axes[name] = clone(table.axisRoles[name] as AxisRole);
	}
	contract.reducers = clone(table.reducers);
	return contract;
}

// -- eval.json files (as euler-view flattens them) -----------------------------------------------

export const RESERVED_TOP_LEVEL_KEYS: ReadonlySet<string> = new Set([
	'metricSet',
	'metric_set',
	'dataset_info',
	'meta',
	'protocol',
	'per_file_metrics',
	'metric_namespace',
	'metricNamespace',
	'task_key',
	'taskKey',
	'metric_producer_key',
	'metricProducerKey',
	'metric_producer_version',
	'metricProducerVersion',
	'metric_source_kind',
	'metricSourceKind',
	'metric_set_metadata',
	'metricSetMetadata'
]);
const METRIC_KEY_RE = /^[a-z0-9_]+$/;

function flatten(node: unknown, name: string, out: { [name: string]: number }): void {
	if (isNumber(node)) {
		if (Number.isFinite(node)) out[name] = node;
		return;
	}
	if (!isRecord(node)) return;
	if (typeof node.type === 'string') return; // typed leaf (distribution)
	for (const [key, value] of Object.entries(node)) {
		if (METRIC_KEY_RE.test(key)) flatten(value, `${name}.${key}`, out);
	}
}

/** Aggregate scalar metrics of a results file: dotted name -> value. */
export function flattenMetrics(payload: unknown): { [name: string]: number } {
	const out: { [name: string]: number } = {};
	for (const [key, value] of Object.entries(rec(payload))) {
		if (RESERVED_TOP_LEVEL_KEYS.has(key) || !METRIC_KEY_RE.test(key)) continue;
		flatten(value, key, out);
	}
	return out;
}

/** item id -> dotted name -> value of a per_file_metrics tree (ids joined with "~"). */
export function flattenPerFile(
	node: unknown,
	path: string[] = [],
	out: { [item: string]: { [name: string]: number } } = {}
): { [item: string]: { [name: string]: number } } {
	if (!isRecord(node)) return out;
	const children = node.children;
	if (isRecord(children)) {
		for (const [key, child] of Object.entries(children)) flattenPerFile(child, [...path, key], out);
	}
	const files = node.files;
	if (Array.isArray(files)) {
		for (const entry of files) {
			if (!isRecord(entry) || typeof entry.id !== 'string') continue;
			const item = [...path, entry.id].join(ITEM_ID_SEPARATOR);
			const values = (out[item] ??= {});
			const metrics = entry.metrics;
			if (isRecord(metrics)) {
				for (const [key, value] of Object.entries(metrics)) {
					if (METRIC_KEY_RE.test(key)) flatten(value, key, values);
				}
			}
		}
	}
	return out;
}

/** Load a parsed results file (eval.json). */
export function loadEvalFile(payload: unknown): EvalFile {
	const raw = rec(payload);
	let metricSet = raw.metricSet;
	if (!isRecord(metricSet)) metricSet = isRecord(raw.metric_set) ? raw.metric_set : {};
	const protocol = isRecord(raw.protocol) ? raw.protocol : null;
	const contract = normalize(protocol, metricSet, raw.meta);
	const perFile = flattenPerFile(raw.per_file_metrics);
	const items = rec(contract.items);
	const seriesRaw = rec(items.series);
	const series: { [name: string]: unknown[] } = {};
	for (const [name, values] of Object.entries(seriesRaw)) {
		if (Array.isArray(values)) series[name] = values;
	}
	const ids = items.ids;
	let itemIds: string[] | null;
	if (isStringArray(ids)) itemIds = [...ids];
	else if (hasOwn(raw, 'per_file_metrics')) itemIds = Object.keys(perFile);
	else itemIds = null;
	return {
		raw,
		namespace: contract.metricSet.metricNamespace,
		axes: axisDeclarations(contract.metricSet.axes) ?? {},
		metricSet: metricSet as Rec,
		protocol,
		contract,
		metrics: flattenMetrics(raw),
		perFile,
		fileIds: Object.keys(perFile),
		series,
		itemIds
	};
}

/** Names with per-item values: per-file metric names and items.series names. */
export function perItemNames(file: EvalFile): Set<string> {
	const names = new Set(Object.keys(file.series));
	for (const values of Object.values(file.perFile)) {
		for (const name of Object.keys(values)) names.add(name);
	}
	return names;
}

/** item id -> value of a per-item name (per-file metrics, then items.series). */
export function perItemValues(file: EvalFile, name: string): { [item: string]: unknown } {
	const values: { [item: string]: unknown } = {};
	for (const [item, metrics] of Object.entries(file.perFile)) {
		if (hasOwn(metrics, name)) values[item] = metrics[name];
	}
	const series = file.series[name];
	const ids = file.itemIds;
	if (Object.keys(values).length === 0 && Array.isArray(series) && ids !== null && series.length === ids.length) {
		ids.forEach((item, index) => {
			values[item] = series[index];
		});
	}
	return values;
}

/**
 * Item set and per-item names of a loaded file for `compare`. The item set is
 * NFC-normalized, or null when unknown -- also when population.manifest is
 * recorded and the ids found in the file do not hash to it.
 */
export function fileEvidence(file: EvalFile, sha: Sha256Hex = sha256Hex): Evidence {
	const ids = file.itemIds;
	let items: Set<string> | null = null;
	if (ids !== null) {
		const normalized = new Set(ids.map((item) => item.normalize('NFC')));
		const manifest = rec(file.contract.population).manifest;
		if (typeof manifest === 'string' && manifestHash(normalized, sha) !== manifest) items = null;
		else items = normalized;
	}
	return { items, perItemNames: perItemNames(file) };
}

// -- the verdict engine (section 8) ---------------------------------------------------------------

export const SEPARATOR = ' · ';
const STEP_SCHEMA = 1;
const STEP_REFERENCES = 2;
const STEP_REVISION = 3;
const STEP_STRATIFICATION = 4;
const STEP_PARAMETERS = 5;
const STEP_POPULATION = 6;

interface Found extends Reason {
	step: number;
}

function severity(tier: Tier): number {
	return TIERS.indexOf(tier);
}

function makeReason(
	tier: Tier,
	code: ReasonCode,
	label: string,
	text: string,
	step: number,
	extra: { key?: string; measure?: Measure; hint?: string | null } = {}
): Found {
	const reason: Found = { tier, code, label, text, step };
	if (extra.key !== undefined) reason.key = extra.key;
	if (extra.measure !== undefined) reason.measure = extra.measure;
	if (extra.hint !== undefined && extra.hint !== null) reason.hint = extra.hint;
	return reason;
}

function publicReason(found: Found): Reason {
	const out: Reason = { tier: found.tier, code: found.code } as Reason;
	if (found.key !== undefined) out.key = found.key;
	out.label = found.label;
	out.text = found.text;
	if (found.measure !== undefined) out.measure = { ...found.measure };
	if (found.hint !== undefined) out.hint = found.hint;
	return out;
}

function finish(found: Found[], shared: number | null): Verdict {
	const reasons = [...found].sort(
		(x, y) =>
			severity(y.tier) - severity(x.tier) ||
			x.step - y.step ||
			compareCodePoints(x.key ?? '', y.key ?? '')
	);
	let tier: Tier = 'same';
	for (const reason of reasons) {
		if (severity(reason.tier) > severity(tier)) tier = reason.tier;
	}
	let label: string;
	if (tier === 'same') {
		label = 'same protocol';
	} else {
		const first = reasons.find((reason) => reason.tier === tier) as Found;
		if (tier === 'paired') label = first.text;
		else if (tier === 'comparable') label = 'comparable' + SEPARATOR + first.text;
		else label = 'not comparable' + SEPARATOR + first.text;
	}
	return { tier, label, shared, reasons: reasons.map(publicReason) };
}

interface Side {
	contract: Rec;
	cell: Cell;
	axesByRole: { [role: string]: string[] };
}

function decomposeCell(contract: Rec, name: string): Side | null {
	const info = rec(contract.metricSet);
	const namespace = info.metricNamespace;
	if (typeof namespace !== 'string') return null;
	const declarations = axisDeclarations(info.axes);
	if (declarations === null) return null;
	let parsed: DecomposedMetric;
	try {
		parsed = decompose(name, namespace, declarations);
	} catch {
		return null;
	}
	const roles: { [role: string]: string[] } = {};
	const axesByRole: { [role: string]: string[] } = {};
	const axisRoles = rec(contract.axes);
	for (const [axisName] of sortedDeclarations(declarations)) {
		const value = parsed.axes[axisName];
		if (value === null || value === undefined) continue;
		let role = rec(axisRoles[axisName]).role;
		if (typeof role !== 'string' || !AXIS_ROLES.includes(role)) role = 'family';
		(roles[role as string] ??= []).push(value);
		(axesByRole[role as string] ??= []).push(axisName);
	}
	return { contract, cell: { name, base: parsed.metric, roles }, axesByRole };
}

function perItem(side: Side, base: string): string {
	const info = rec(side.contract.metricSet);
	const namespace = info.metricNamespace as string;
	const declarations = axisDeclarations(info.axes) ?? {};
	const parsed = decompose(side.cell.name, namespace, declarations);
	const reductionAxes = new Set((side.axesByRole.reduction ?? []).slice(0, 1));
	const parts = [namespace];
	for (const [axisName] of sortedDeclarations(declarations)) {
		const value = parsed.axes[axisName];
		if (value !== null && value !== undefined && !reductionAxes.has(axisName)) parts.push(value);
	}
	parts.push(base);
	return parts.join('.');
}

/** The per-item name of an aggregate metric name (section 4.11). */
export function perItemName(contract: Contract | Rec, metricName: string): string | null {
	const side = decomposeCell(contract as Rec, metricName);
	return side === null ? null : perItem(side, side.cell.base);
}

function definitionOf(contract: Rec, base: string): Rec | null {
	const definition = rec(contract.metrics)[base];
	return isRecord(definition) ? definition : null;
}

function unique<T>(values: T[]): T[] {
	return [...new Set(values)];
}

function mergedDefinition(a: Rec | null, b: Rec | null): Rec {
	if (a === null || b === null) return { ...(a ?? b ?? {}) };
	const merged: Rec = { ...a };
	const reqA = a.requires;
	const reqB = b.requires;
	if (Array.isArray(reqA) && Array.isArray(reqB)) merged.requires = unique([...reqA, ...reqB]);
	else delete merged.requires;
	const depA = Array.isArray(a.dependsOn) ? a.dependsOn : ['*'];
	const depB = Array.isArray(b.dependsOn) ? b.dependsOn : ['*'];
	merged.dependsOn = unique([...depA, ...depB]);
	const poolA = isRecord(a.pool) ? a.pool : { form: 'none' };
	const poolB = isRecord(b.pool) ? b.pool : { form: 'none' };
	merged.pool = jsonEqual(poolA, poolB) ? poolA : { form: 'none' };
	if (!jsonEqual(get(a, 'reducer'), get(b, 'reducer'))) delete merged.reducer;
	return merged;
}

function relevant(
	key: string,
	descriptor: MergedDescriptor,
	cell: Cell,
	definition: Rec,
	contextA: Context,
	contextB: Context,
	reference = false
): boolean {
	const depends = Array.isArray(definition.dependsOn) ? definition.dependsOn : ['*'];
	if (!globAny(depends, key)) return false;
	if (!mergedAffects(descriptor, cell)) return false;
	if (reference) {
		const requires = definition.requires;
		if (Array.isArray(requires) && !requires.includes(key.slice('reference.'.length))) return false;
	}
	return !mergedSkipped(descriptor, contextA, contextB);
}

/** Judge metric cell `metricName` of normalized contracts `a` and `b` (section 8). */
export function compare(a: Contract | Rec, b: Contract | Rec, metricName: string, options: CompareOptions = {}): Verdict {
	const ca = a as Rec;
	const cb = b as Rec;
	if (!(isSupported(ca) && isSupported(cb))) return unsupported(ca, cb);

	// 8.1 cell
	const sideA = decomposeCell(ca, metricName);
	const sideB = decomposeCell(cb, metricName);
	if (sideA === null || sideB === null) {
		return finish([makeReason('incomparable', 'schema', metricName, 'metric name does not decompose', STEP_SCHEMA)], null);
	}
	if (sideA.cell.base !== sideB.cell.base || !jsonEqual(sideA.cell.roles, sideB.cell.roles)) {
		return finish(
			[makeReason('incomparable', 'schema', metricName, 'metric name decomposes differently', STEP_SCHEMA)],
			null
		);
	}
	const cell = sideA.cell;
	const base = cell.base;
	const defA = definitionOf(ca, base);
	const defB = definitionOf(cb, base);
	const definition = mergedDefinition(defA, defB);
	const contextA: Context = [rec(ca.parameters), rec(ca.observations)];
	const contextB: Context = [rec(cb.parameters), rec(cb.observations)];
	const descriptorsA = rec(ca.descriptors);
	const descriptorsB = rec(cb.descriptors);
	const reasons: Found[] = [];

	// 8.5 references
	const refsA = rec(ca.references);
	const refsB = rec(cb.references);
	for (const role of sortedCodePoints(new Set([...Object.keys(refsA), ...Object.keys(refsB)]))) {
		const key = `reference.${role}`;
		const descriptor = mergeDescriptors(key, descriptorsA[key], descriptorsB[key]);
		if (!relevant(key, descriptor, cell, definition, contextA, contextB, true)) continue;
		reasons.push(...referenceReasons(key, descriptor, refsA[role], refsB[role]));
	}

	// 8.2 revision
	const revA = defA !== null ? defA.rev : null;
	const revB = defB !== null ? defB.rev : null;
	const hasA = validRev(revA);
	const hasB = validRev(revB);
	if (hasA && hasB && revA !== revB) {
		reasons.push(
			makeReason(
				'incomparable',
				'revision',
				base,
				`${base} definition revised (rev ${formatNumber(revA)} vs ${formatNumber(revB)})`,
				STEP_REVISION
			)
		);
	} else if (hasA !== hasB) {
		reasons.push(makeReason('incomparable', 'not_recorded', base, `${base} revision not recorded`, STEP_REVISION));
	}

	// 8.3 stratification
	const strataA = sideA.axesByRole.stratum ?? [];
	const strataB = sideB.axesByRole.stratum ?? [];
	cellValues(cell, 'stratum').forEach((value, index) => {
		reasons.push(...stratumReasons(ca, cb, strataA[index] as string, strataB[index] as string, value));
	});

	// 8.6 parameters and observations
	const [paramsA, obsA] = contextA;
	const [paramsB, obsB] = contextB;
	const keys = sortedCodePoints(
		new Set([...Object.keys(paramsA), ...Object.keys(obsA), ...Object.keys(paramsB), ...Object.keys(obsB)])
	);
	for (const key of keys) {
		const descriptor = mergeDescriptors(key, descriptorsA[key], descriptorsB[key]);
		if (descriptor.compare === 'ignore' || descriptor.compare === 'pair') continue;
		if (!relevant(key, descriptor, cell, definition, contextA, contextB)) continue;
		const valueA = hasOwn(paramsA, key) ? paramsA[key] : hasOwn(obsA, key) ? obsA[key] : MISSING;
		const valueB = hasOwn(paramsB, key) ? paramsB[key] : hasOwn(obsB, key) ? obsB[key] : MISSING;
		const reason = parameterReason(key, descriptor, valueA, valueB);
		if (reason !== null) reasons.push(reason);
	}

	// 8.7 population
	const [shared, reason] = population(ca, cb, sideA, sideB, definition, options);
	if (reason !== null) reasons.push(reason);
	return finish(reasons, shared);
}


function referenceReasons(key: string, descriptor: MergedDescriptor, refA: unknown, refB: unknown): Found[] {
	const label = descriptor.label;
	const identityA = isRecord(refA) ? refA.identity : null;
	const identityB = isRecord(refB) ? refB.identity : null;
	if (typeof identityA !== 'string' || typeof identityB !== 'string') {
		return [
			makeReason('incomparable', 'not_recorded', label, `${label} not recorded`, STEP_REFERENCES, {
				key,
				hint: descriptor.hint
			})
		];
	}
	if (identityA === identityB) return [];
	const viewA = (refA as Rec).view;
	const viewB = (refB as Rec).view;
	if (isRecord(viewA) && isRecord(viewB) && typeof viewA.source === 'string' && viewA.source === viewB.source) {
		const frame = viewA.source;
		const d = measure('region', { frame, box: get(viewA, 'region') }, { frame, box: get(viewB, 'region') });
		if (d !== null && d > 0) {
			return [
				makeReason('comparable', 'domain', label, phrase('region', `${label} region`, null, null, d), STEP_REFERENCES, {
					key,
					measure: { kind: 'region', d }
				})
			];
		}
	}
	return [
		makeReason('incomparable', 'reference', label, `${label} differs`, STEP_REFERENCES, { key, hint: descriptor.hint })
	];
}

function stratumReasons(a: Rec, b: Rec, axisA: string, axisB: string, value: string): Found[] {
	const sidA = rec(rec(a.axes)[axisA]).stratification;
	const sidB = rec(rec(b.axes)[axisB]).stratification;
	const stratA = typeof sidA === 'string' ? rec(a.stratifications)[sidA] : null;
	const stratB = typeof sidB === 'string' ? rec(b.stratifications)[sidB] : null;
	let label: string | null = null;
	for (const strat of [stratA, stratB]) {
		if (isRecord(strat) && typeof strat.label === 'string') {
			label = strat.label;
			break;
		}
	}
	const shown = label || axisA;
	if (!isRecord(stratA) || !isRecord(stratB)) {
		return [makeReason('incomparable', 'not_recorded', shown, 'bins not recorded', STEP_STRATIFICATION)];
	}
	if (jsonEqual(identityProjection(stratA), identityProjection(stratB))) return [];
	return [makeReason('incomparable', 'stratification', shown, binsText(stratA, stratB, value), STEP_STRATIFICATION)];
}

function binsText(stratA: Rec, stratB: Rec, value: string): string {
	const rangeA = binRange(stratA, value);
	const rangeB = binRange(stratB, value);
	if (rangeA === null || rangeB === null) return `bins differ (${value})`;
	const textA = withUnit(`${edgeText(rangeA[0], true)}–${edgeText(rangeA[1], false)}`, stratA);
	const textB = withUnit(`${edgeText(rangeB[0], true)}–${edgeText(rangeB[1], false)}`, stratB);
	if (textA !== textB) return `bins differ (${value} ${textA} vs ${textB})`;
	const edgesA = withUnit(edgesText(stratA), stratA);
	const edgesB = withUnit(edgesText(stratB), stratB);
	if (edgesA !== edgesB) return `bins differ (${value}: edges ${edgesA} vs ${edgesB})`;
	return `bins differ (${value})`;
}

/** An edge in phrases: 3 significant digits; an open end is −∞ / ∞. */
function edgeText(value: number | null, low: boolean): string {
	if (value === null) return low ? '−∞' : '∞';
	return formatSig3(value);
}

function edgesText(stratification: Rec): string {
	const edges = stratification.edges as (number | null)[];
	return '[' + edges.map((edge, index) => edgeText(edge, index === 0)).join(', ') + ']';
}

function withUnit(text: string, stratification: Rec): string {
	const unit = stratification.unit;
	return typeof unit === 'string' && unit ? `${text} ${unit}` : text;
}

function parameterReason(
	key: string,
	descriptor: MergedDescriptor,
	rawA: unknown,
	rawB: unknown
): Found | null {
	const label = descriptor.label;
	let valueA = rawA;
	let valueB = rawB;
	if (valueA === MISSING && descriptor.hasDefault) valueA = descriptor.defaultValue;
	if (valueB === MISSING && descriptor.hasDefault) valueB = descriptor.defaultValue;
	if (valueA === MISSING || valueB === MISSING) {
		return makeReason('incomparable', 'not_recorded', label, `${label} not recorded`, STEP_PARAMETERS, {
			key,
			hint: descriptor.hint
		});
	}
	if (jsonEqual(valueA, valueB)) return null;
	if (descriptor.compare === 'measure') {
		const d = measure(descriptor.observe, valueA, valueB);
		if (d !== null) {
			if (d <= 0) return null;
			const kind = descriptor.observe as ObserveKind;
			return makeReason('comparable', 'domain', label, phrase(kind, label, valueA, valueB, d), STEP_PARAMETERS, {
				key,
				measure: { kind, d }
			});
		}
	}
	return makeReason('incomparable', 'parameter', label, `${label} differs`, STEP_PARAMETERS, {
		key,
		hint: descriptor.hint
	});
}

function population(
	a: Rec,
	b: Rec,
	sideA: Side,
	sideB: Side,
	definition: Rec,
	options: CompareOptions
): [number | null, Found | null] {
	const manifestA = rec(a.population).manifest;
	const manifestB = rec(b.population).manifest;
	if (typeof manifestA === 'string' && manifestA && manifestA === manifestB) return [null, null];
	if (options.itemsA == null || options.itemsB == null) {
		return [
			null,
			makeReason('incomparable', 'population_unknown', 'files', 'evaluated files not recorded', STEP_POPULATION)
		];
	}
	const setA = new Set(options.itemsA);
	const setB = new Set(options.itemsB);
	if (setA.size === setB.size && [...setA].every((item) => setB.has(item))) return [null, null];
	let shared = 0;
	for (const item of setA) if (setB.has(item)) shared += 1;
	if (shared === 0) {
		return [0, makeReason('incomparable', 'population_disjoint', 'files', 'no shared files', STEP_POPULATION)];
	}
	if (pairable(a, b, sideA, sideB, definition, options)) {
		return [
			shared,
			makeReason('paired', 'population_paired', 'files', `compared on ${shared} shared files`, STEP_POPULATION)
		];
	}
	const total = Math.max(setA.size, setB.size);
	return [
		shared,
		makeReason(
			'comparable',
			'population_partial',
			'files',
			`computed on different files: ${shared} of ${total} shared`,
			STEP_POPULATION
		)
	];
}

/** The recompute op of the cell's reducer: its reduction-axis value's, else the metric's `reducer`. */
function recomputeOp(contract: Rec, side: Side, definition: Rec): string | null {
	let value = cellValue(side.cell, 'reduction');
	if (value === null) {
		const implicit = definition.reducer;
		if (typeof implicit !== 'string') return null;
		value = implicit;
	}
	const reducer = rec(contract.reducers)[value];
	const op = isRecord(reducer) ? reducer.recompute : null;
	return op === 'mean' || op === 'median' || op === 'pool' ? op : null;
}

function pairable(a: Rec, b: Rec, sideA: Side, sideB: Side, definition: Rec, options: CompareOptions): boolean {
	const op = recomputeOp(a, sideA, definition);
	if (op === null || op !== recomputeOp(b, sideB, definition)) return false;
	if (options.perItemA == null || options.perItemB == null) return false;
	const namesA = new Set(options.perItemA);
	const namesB = new Set(options.perItemB);
	if (!namesA.has(perItem(sideA, sideA.cell.base)) || !namesB.has(perItem(sideB, sideB.cell.base))) return false;
	if (op === 'pool') {
		const pool = definition.pool;
		if (!isRecord(pool) || !(pool.form === 'mean' || pool.form === 'rms')) return false;
		const weight = pool.weight;
		if (typeof weight !== 'string' || !weight) return false;
		if (!namesA.has(perItem(sideA, weight)) || !namesB.has(perItem(sideB, weight))) return false;
	}
	return true;
}

function unsupported(a: Rec, b: Rec): Verdict {
	const paramsA = rec(a.parameters);
	const paramsB = rec(b.parameters);
	const versions = [a.schemaVersion, b.schemaVersion].filter((version) => {
		const major = schemaMajor(version);
		return major === null || !SUPPORTED_MAJORS.includes(major);
	});
	const shown = versions.length > 0 && typeof versions[0] === 'string' ? versions[0] : '?';
	const reasons: Found[] = [
		makeReason('same', 'schema', 'schema', `compared by parameters only (schema ${shown} not supported)`, STEP_SCHEMA)
	];
	for (const key of sortedCodePoints(new Set([...Object.keys(paramsA), ...Object.keys(paramsB)]))) {
		if (!hasOwn(paramsA, key) || !hasOwn(paramsB, key)) {
			reasons.push(makeReason('incomparable', 'schema', key, `${key} not recorded`, STEP_SCHEMA, { key }));
		} else if (!jsonEqual(paramsA[key], paramsB[key])) {
			reasons.push(makeReason('incomparable', 'schema', key, `${key} differs`, STEP_SCHEMA, { key }));
		}
	}
	return finish(reasons, null);
}

/** `compare` with item sets and per-item names taken from two loaded files. */
export function compareFiles(fileA: EvalFile, fileB: EvalFile, metricName: string, sha: Sha256Hex = sha256Hex): Verdict {
	const evidenceA = fileEvidence(fileA, sha);
	const evidenceB = fileEvidence(fileB, sha);
	return compare(fileA.contract, fileB.contract, metricName, {
		itemsA: evidenceA.items,
		itemsB: evidenceB.items,
		perItemA: evidenceA.perItemNames,
		perItemB: evidenceB.perItemNames
	});
}

// -- recompute on shared items (sections 3.7, 3.8) -----------------------------------------------

/**
 * Recompute an aggregate from per-item values on `items` (all when omitted).
 * For `pool`, `form` is the metric's pool.form and `weights` its weight series.
 */
export function recompute(
	op: string,
	values: { [item: string]: unknown },
	items?: Iterable<string> | null,
	weights?: { [item: string]: unknown } | null,
	form?: string | null
): number | null {
	const allowed = items == null ? null : new Set(items);
	const keys = sortedCodePoints(Object.keys(values).filter((key) => allowed === null || allowed.has(key)));
	const pairs: [number, number][] = [];
	for (const key of keys) {
		const value = values[key];
		if (!isNumber(value) || !Number.isFinite(value)) continue;
		if (op === 'pool') {
			const weight = weights ? weights[key] : undefined;
			if (!isNumber(weight) || !Number.isFinite(weight) || weight === 0) continue;
			pairs.push([value, weight]);
		} else {
			pairs.push([value, 1]);
		}
	}
	if (pairs.length === 0) return null;
	if (op === 'mean') {
		let total = 0;
		for (const [value] of pairs) total += value;
		return total / pairs.length;
	}
	if (op === 'median') {
		const ordered = pairs.map(([value]) => value).sort((x, y) => x - y);
		const middle = Math.floor(ordered.length / 2);
		if (ordered.length % 2) return ordered[middle] as number;
		return ((ordered[middle - 1] as number) + (ordered[middle] as number)) / 2;
	}
	if (op === 'pool' && (form === 'mean' || form === 'rms')) {
		let numerator = 0;
		let denominator = 0;
		for (const [value, weight] of pairs) {
			numerator += weight * (form === 'rms' ? value * value : value);
			denominator += weight;
		}
		const result = numerator / denominator;
		return form === 'rms' ? Math.sqrt(result) : result;
	}
	return null;
}

/** Recompute aggregate `metricName` of a loaded file on `items`. */
export function recomputeCell(
	contract: Contract | Rec,
	file: EvalFile,
	metricName: string,
	items?: Iterable<string> | null
): number | null {
	const side = decomposeCell(contract as Rec, metricName);
	if (side === null) return null;
	const definition = definitionOf(contract as Rec, side.cell.base) ?? {};
	const op = recomputeOp(contract as Rec, side, definition);
	if (op === null) return null;
	const values = perItemValues(file, perItem(side, side.cell.base));
	let weights: { [item: string]: unknown } | null = null;
	let form: string | null = null;
	if (op === 'pool') {
		const pool = rec(definition.pool);
		form = typeof pool.form === 'string' ? pool.form : null;
		const weight = pool.weight;
		if (typeof weight !== 'string') return null;
		weights = perItemValues(file, perItem(side, weight));
	}
	return recompute(op, values, items, weights, form);
}
