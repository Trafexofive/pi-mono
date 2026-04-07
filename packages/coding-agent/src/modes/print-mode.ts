/**
 * Print mode (single-shot): Send prompts, output result, exit.
 *
 * Used for:
 * - `pi -p "prompt"` - text output
 * - `pi --mode json "prompt"` - JSON event stream
 */

import type { AssistantMessage, ImageContent } from "@mariozechner/pi-ai";
import type { AgentSessionRuntime } from "../core/agent-session-runtime.js";
import { flushRawStdout, writeRawStdout } from "../core/output-guard.js";

/**
 * Options for print mode.
 */
export interface PrintModeOptions {
	/** Output mode: "text" for final response only, "json" for all events */
	mode: "text" | "json";
	/** Array of additional prompts to send after initialMessage */
	messages?: string[];
	/** First message to send (may contain @file content) */
	initialMessage?: string;
	/** Images to attach to the initial message */
	initialImages?: ImageContent[];
}

/**
 * Run in print (single-shot) mode.
 * Sends prompts to the agent and outputs the result.
 */
export async function runPrintMode(runtimeHost: AgentSessionRuntime, options: PrintModeOptions): Promise<number> {
	const { mode, messages = [], initialMessage, initialImages } = options;
	let exitCode = 0;
	let session = runtimeHost.session;
	let unsubscribe: (() => void) | undefined;

	const rebindSession = async (): Promise<void> => {
		session = runtimeHost.session;
		await session.bindExtensions({
			commandContextActions: {
				waitForIdle: () => session.agent.waitForIdle(),
				newSession: async (newSessionOptions) => {
					const result = await runtimeHost.newSession(newSessionOptions);
					if (!result.cancelled) {
						await rebindSession();
					}
					return result;
				},
				fork: async (entryId) => {
					const result = await runtimeHost.fork(entryId);
					if (!result.cancelled) {
						await rebindSession();
					}
					return { cancelled: result.cancelled };
				},
				navigateTree: async (targetId, navigateOptions) => {
					const result = await session.navigateTree(targetId, {
						summarize: navigateOptions?.summarize,
						customInstructions: navigateOptions?.customInstructions,
						replaceInstructions: navigateOptions?.replaceInstructions,
						label: navigateOptions?.label,
					});
					return { cancelled: result.cancelled };
				},
				switchSession: async (sessionPath) => {
					const result = await runtimeHost.switchSession(sessionPath);
					if (!result.cancelled) {
						await rebindSession();
					}
					return result;
				},
				reload: async () => {
					await session.reload();
				},
			},
			onError: (err) => {
				console.error(`Extension error (${err.extensionPath}): ${err.error}`);
			},
		});

		unsubscribe?.();
		unsubscribe = session.subscribe((event) => {
			if (mode === "json") {
				writeRawStdout(`${JSON.stringify(event)}\n`);
			}
		});
	};

	// ── Retry helpers for rate-limit / upstream errors ──────────────
	const RETRYABLE =
		/provider.?returned.?error|rate.?(limit|increased)|429|too many requests|overloaded|503|502|upstream.?connect|timed? ?out/i;

	const isRetryable = (msg: string) => RETRYABLE.test(msg);

	const promptWithRetry = async (prompt: string, opts?: { images?: ImageContent[] }): Promise<boolean> => {
		const maxAttempts = 50;
		for (let attempt = 1; attempt <= maxAttempts; attempt++) {
			try {
				await session.prompt(prompt, opts);

				// Check if the prompt ended with an error state
				const state = session.state;
				const last = state.messages[state.messages.length - 1];
				if (last?.role === "assistant" && last.stopReason === "error") {
					const errMsg = (last as AssistantMessage).errorMessage ?? "";
					if (!isRetryable(errMsg) || attempt === maxAttempts) {
						return false; // non-retryable error or exhausted retries
					}
					const delayMs = 2000 * 2 ** (attempt - 1);
					console.error(
						`[retry ${attempt}/${maxAttempts}] rate-limit/upstream error, waiting ${delayMs / 1000}s: ${errMsg.substring(0, 120)}`,
					);
					await new Promise((r) => setTimeout(r, delayMs));
					continue;
				}
				return true;
			} catch (err: unknown) {
				const msg = err instanceof Error ? err.message : String(err);
				if (!isRetryable(msg) || attempt === maxAttempts) {
					throw err;
				}
				const delayMs = 2000 * 2 ** (attempt - 1);
				console.error(
					`[retry ${attempt}/${maxAttempts}] ${msg.substring(0, 120)} — retrying in ${delayMs / 1000}s`,
				);
				await new Promise((r) => setTimeout(r, delayMs));
			}
		}
		return false;
	};

	try {
		if (mode === "json") {
			const header = session.sessionManager.getHeader();
			if (header) {
				writeRawStdout(`${JSON.stringify(header)}\n`);
			}
		}

		await rebindSession();

		if (initialMessage) {
			const ok = await promptWithRetry(initialMessage, { images: initialImages });
			if (!ok) {
				const state = session.state;
				const last = state.messages[state.messages.length - 1];
				if (last?.role === "assistant") {
					console.error((last as AssistantMessage).errorMessage ?? "request failed");
				}
				return 1;
			}
		}

		for (const message of messages) {
			const ok = await promptWithRetry(message);
			if (!ok) {
				const state = session.state;
				const last = state.messages[state.messages.length - 1];
				if (last?.role === "assistant") {
					console.error((last as AssistantMessage).errorMessage ?? "request failed");
				}
				return 1;
			}
		}

		if (mode === "text") {
			const state = session.state;
			const lastMessage = state.messages[state.messages.length - 1];

			if (lastMessage?.role === "assistant") {
				const assistantMsg = lastMessage as AssistantMessage;
				if (assistantMsg.stopReason === "error" || assistantMsg.stopReason === "aborted") {
					console.error(assistantMsg.errorMessage || `Request ${assistantMsg.stopReason}`);
					exitCode = 1;
				} else {
					for (const content of assistantMsg.content) {
						if (content.type === "text") {
							writeRawStdout(`${content.text}\n`);
						}
					}
				}
			}
		}

		return exitCode;
	} catch (error: unknown) {
		console.error(error instanceof Error ? error.message : String(error));
		return 1;
	} finally {
		unsubscribe?.();
		await runtimeHost.dispose();
		await flushRawStdout();
	}
}
