// useChat — Ink hook for conversational AI via the Xcel assistant API.
// Manages message history, SSE streaming, and confirmation flow.

import { useState, useCallback, useRef } from "react";
import { streamChat, confirmAction, type SSEEvent, type ApiClientConfig } from "./api-client.js";
import type { WizardState } from "../sprites/wizard/wizard-sprite.js";

export interface ChatMessage {
    role: "user" | "assistant" | "system";
    content: string;
    toolCalls?: { name: string; input: Record<string, unknown> }[];
    toolResults?: { name: string; output: Record<string, unknown> }[];
}

export interface PendingConfirmation {
    confirmationId: string;
    toolName: string;
    toolArgs: Record<string, unknown>;
}

export interface UseChatReturn {
    messages: ChatMessage[];
    wizardState: WizardState;
    wizardMessage: string;
    pendingConfirmation: PendingConfirmation | null;
    conversationId: string | null;
    sendMessage: (text: string) => Promise<void>;
    confirm: (approved: boolean) => Promise<void>;
    isStreaming: boolean;
}

export function useChat(config: ApiClientConfig): UseChatReturn {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [wizardState, setWizardState] = useState<WizardState>("idle");
    const [wizardMessage, setWizardMessage] = useState("Ask me anything — I'll guide you through setup.");
    const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const conversationIdRef = useRef<string | null>(null);

    const sendMessage = useCallback(async (text: string) => {
        // Add user message
        setMessages((prev) => [...prev, { role: "user", content: text }]);
        setWizardState("thinking");
        setWizardMessage("Thinking...");
        setIsStreaming(true);
        setPendingConfirmation(null);

        let assistantContent = "";
        const toolCalls: ChatMessage["toolCalls"] = [];
        const toolResults: ChatMessage["toolResults"] = [];

        try {
            for await (const event of streamChat(config, text, conversationIdRef.current ?? undefined)) {
                switch (event.type) {
                    case "meta":
                        if (event.conversation_id) {
                            conversationIdRef.current = event.conversation_id;
                        }
                        break;

                    case "token":
                        assistantContent += event.content ?? "";
                        // Show a truncated preview in the wizard line
                        const preview = assistantContent.length > 80
                            ? "..." + assistantContent.slice(-77)
                            : assistantContent;
                        setWizardMessage(preview.replace(/\n/g, " "));
                        break;

                    case "tool_call":
                        if (event.name) {
                            toolCalls.push({ name: event.name, input: event.input ?? {} });
                            setWizardMessage(`Using ${event.name}...`);
                        }
                        break;

                    case "tool_result":
                        if (event.name) {
                            toolResults.push({ name: event.name, output: event.output ?? {} });
                        }
                        break;

                    case "confirmation_required":
                        setPendingConfirmation({
                            confirmationId: event.confirmation_id!,
                            toolName: event.tool_name!,
                            toolArgs: event.tool_args ?? {},
                        });
                        setWizardState("idle");
                        setWizardMessage(`Confirm: ${event.tool_name}? (y/n)`);
                        break;

                    case "error":
                        setWizardState("error");
                        setWizardMessage(event.message ?? "Something went wrong");
                        break;

                    case "done":
                        break;
                }
            }

            // Add the assistant message
            if (assistantContent) {
                setMessages((prev) => [
                    ...prev,
                    { role: "assistant", content: assistantContent, toolCalls, toolResults },
                ]);
            }

            if (!pendingConfirmation) {
                setWizardState("success");
                setWizardMessage("Ready — ask me anything or follow the steps above.");
            }
        } catch (err) {
            setWizardState("error");
            const msg = err instanceof Error ? err.message : "Connection failed";
            setWizardMessage(msg);
            setMessages((prev) => [...prev, { role: "system", content: `Error: ${msg}` }]);
        } finally {
            setIsStreaming(false);
        }
    }, [config, pendingConfirmation]);

    const confirm = useCallback(async (approved: boolean) => {
        if (!pendingConfirmation) return;

        setWizardState("thinking");
        setWizardMessage(approved ? "Executing..." : "Cancelled.");
        setIsStreaming(true);

        let content = "";
        try {
            for await (const event of confirmAction(config, pendingConfirmation.confirmationId, approved)) {
                if (event.type === "token") {
                    content += event.content ?? "";
                } else if (event.type === "error") {
                    setWizardState("error");
                    setWizardMessage(event.message ?? "Confirmation failed");
                }
            }

            if (content) {
                setMessages((prev) => [...prev, { role: "assistant", content }]);
            }

            setPendingConfirmation(null);
            setWizardState(approved ? "success" : "idle");
            setWizardMessage(approved ? "Action completed." : "Action cancelled. What next?");
        } catch (err) {
            setWizardState("error");
            setWizardMessage(err instanceof Error ? err.message : "Confirmation failed");
        } finally {
            setIsStreaming(false);
        }
    }, [config, pendingConfirmation]);

    return {
        messages,
        wizardState,
        wizardMessage,
        pendingConfirmation,
        conversationId: conversationIdRef.current,
        sendMessage,
        confirm,
        isStreaming,
    };
}
