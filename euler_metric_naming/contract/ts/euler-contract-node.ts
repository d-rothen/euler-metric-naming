/**
 * Node-only wiring of the contract's hashes to node:crypto.
 *
 * euler-contract.ts ships a pure-TS SHA-256 and works without this file (in
 * the browser too); servers that hash large population manifests may prefer
 * node:crypto. Same results, verified by conformance.test.ts.
 */
import { createHash } from 'node:crypto';

import {
	compareFiles as compareFilesWith,
	definitionKey as definitionKeyWith,
	fileEvidence as fileEvidenceWith,
	hashValue as hashValueWith,
	manifestHash as manifestHashWith
} from './euler-contract.ts';
import type { Contract, EvalFile, Evidence, Verdict } from './euler-contract.ts';

export * from './euler-contract.ts';

/** SHA-256 hex digest through node:crypto. */
export function nodeSha256Hex(bytes: Uint8Array): string {
	return createHash('sha256').update(bytes).digest('hex');
}

export function nodeHashValue(value: unknown): string {
	return hashValueWith(value, nodeSha256Hex);
}

export function nodeManifestHash(ids: Iterable<string>): string {
	return manifestHashWith(ids, nodeSha256Hex);
}

export function nodeDefinitionKey(contract: Contract): string {
	return definitionKeyWith(contract, nodeSha256Hex);
}

export function nodeFileEvidence(file: EvalFile): Evidence {
	return fileEvidenceWith(file, nodeSha256Hex);
}

export function nodeCompareFiles(fileA: EvalFile, fileB: EvalFile, metricName: string): Verdict {
	return compareFilesWith(fileA, fileB, metricName, nodeSha256Hex);
}
