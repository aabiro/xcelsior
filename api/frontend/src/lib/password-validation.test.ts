import { describe, expect, it } from "vitest";
import {
  PASSWORD_MAX_LENGTH,
  getPasswordValidation,
  passwordsMatch,
} from "@/lib/password-validation";

describe("password validation", () => {
  it("accepts passwords that satisfy every rule", () => {
    expect(getPasswordValidation("StrongPass123!").isValid).toBe(true);
  });

  it("rejects unsupported symbols", () => {
    const result = getPasswordValidation("StrongPass123?");

    expect(result.hasSupportedSymbol).toBe(false);
    expect(result.hasUnsupportedCharacter).toBe(true);
    expect(result.isValid).toBe(false);
  });

  it("rejects passwords longer than the allowed maximum", () => {
    const tooLongPassword = `Aa1!${"b".repeat(PASSWORD_MAX_LENGTH)}`;

    expect(getPasswordValidation(tooLongPassword).hasValidLength).toBe(false);
  });

  it("only treats non-empty matching confirmation fields as valid", () => {
    expect(passwordsMatch("StrongPass123!", "")).toBe(false);
    expect(passwordsMatch("StrongPass123!", "StrongPass123!")).toBe(true);
  });
});
