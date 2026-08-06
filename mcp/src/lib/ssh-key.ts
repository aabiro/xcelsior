/**
 * What did the user actually paste?
 *
 * `register_ssh_key` is the one tool where the *wrong* argument is a secret.
 * A user asked for "my SSH key" hands over `id_ed25519` about as often as
 * `id_ed25519.pub`, and a model relaying it has no reason to know the
 * difference. So the distinction is made here, before anything is sent, and it
 * is a pure function so the decision can be tested exhaustively rather than
 * inferred from a handler that also does network I/O.
 *
 * The plan's line is "the private key never exists server-side and never enters
 * model context". The second half is already lost by the time this runs — the
 * key is in the tool argument. What is still salvageable is telling the user
 * immediately, while they can still rotate, rather than letting the API answer
 * `400 invalid key` and leaving them to guess why.
 */

/** OpenSSH and PEM private-key envelopes, including the encrypted variants. */
const PRIVATE_KEY_ENVELOPE = /-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY(?: BLOCK)?-----/;

/**
 * PuTTY's format, which carries no PEM envelope at all.
 *
 * Worth its own pattern: a `.ppk` is the file a Windows user is most likely to
 * have, it is a private key, and matching only on `BEGIN ... PRIVATE KEY`
 * would wave it straight through to the "not a public key" branch — whose
 * message asks them to paste their key file, which is what they just did.
 */
const PUTTY_PRIVATE_KEY = /^PuTTY-User-Key-File-\d+:/m;

/**
 * The key types OpenSSH will accept in `authorized_keys`.
 *
 * `ssh-dss` is included because a key already in use should register rather
 * than be refused by a client-side list the server does not share; whether
 * DSA is acceptable is the server's ruling, and it makes it.
 */
const PUBLIC_KEY_LINE =
  /^(?:ssh-ed25519|ssh-rsa|ssh-dss|ecdsa-sha2-[a-z0-9-]+|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-[a-z0-9-]+@openssh\.com)\s+[A-Za-z0-9+/]+={0,3}(?:\s|$)/;

export type SshKeyVerdict = "public" | "private" | "unrecognized";

export interface SshKeyInspection {
  verdict: SshKeyVerdict;
  /** The single key line, trimmed of surrounding blank lines. Empty unless public. */
  key: string;
  /** What to tell the user. Empty when the verdict is `public`. */
  message: string;
}

export function inspectSshKeyInput(raw: string): SshKeyInspection {
  const text = (raw ?? "").trim();

  if (PRIVATE_KEY_ENVELOPE.test(text) || PUTTY_PRIVATE_KEY.test(text)) {
    return {
      verdict: "private",
      key: "",
      message:
        "That is a PRIVATE key, and it was not sent anywhere. Treat it as " +
        "compromised — it has been through this conversation — and generate a " +
        "new pair. Register the .pub file instead: one line beginning with " +
        "'ssh-ed25519', 'ssh-rsa' or 'ecdsa-sha2-'.",
    };
  }

  // A pasted .pub file is one line, but it arrives with trailing newlines, and
  // occasionally with a leading comment line from a shell transcript.
  const line = text.split(/\r?\n/).map((l) => l.trim()).find((l) => PUBLIC_KEY_LINE.test(l));
  if (!line) {
    return {
      verdict: "unrecognized",
      key: "",
      message:
        "That does not look like an SSH public key. Paste the contents of your " +
        ".pub file — a single line beginning with a key type such as " +
        "'ssh-ed25519', followed by the key data.",
    };
  }

  return { verdict: "public", key: line, message: "" };
}
