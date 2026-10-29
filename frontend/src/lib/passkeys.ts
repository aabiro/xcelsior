export function describePasskeyRegistrationError(err: unknown): string {
  const duplicateMessage =
    "This passkey is already added to your account. Use it to sign in or add a different device.";

  if (err instanceof DOMException) {
    if (err.name === "NotAllowedError") {
      return "Passkey registration was cancelled.";
    }
    if (err.name === "InvalidStateError") {
      return duplicateMessage;
    }
  }

  const message = err instanceof Error ? err.message : "";
  if (/already added|already registered|already exists|credential.+already/i.test(message)) {
    return duplicateMessage;
  }
  return message || "Failed to register passkey";
}
