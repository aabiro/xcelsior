export const PASSWORD_MIN_LENGTH = 8;
export const PASSWORD_MAX_LENGTH = 64;
export const PASSWORD_SUPPORTED_SYMBOLS = "!@#$%^*-_+=";

const PASSWORD_SUPPORTED_SYMBOL_SET = new Set(PASSWORD_SUPPORTED_SYMBOLS.split(""));
const unicodeNumberPattern = /\p{N}/u;

function isLetter(char: string) {
  return char.toLocaleLowerCase() !== char.toLocaleUpperCase();
}

function isNumber(char: string) {
  return unicodeNumberPattern.test(char);
}

export interface PasswordValidationResult {
  hasValidLength: boolean;
  hasLetter: boolean;
  hasNumber: boolean;
  hasSupportedSymbol: boolean;
  hasUnsupportedCharacter: boolean;
  isValid: boolean;
}

export function getPasswordValidation(password: string): PasswordValidationResult {
  let hasLetter = false;
  let hasNumber = false;
  let hasSupportedSymbol = false;
  let hasUnsupportedCharacter = false;

  for (const char of Array.from(password)) {
    if (isLetter(char)) {
      hasLetter = true;
      continue;
    }

    if (isNumber(char)) {
      hasNumber = true;
      continue;
    }

    if (PASSWORD_SUPPORTED_SYMBOL_SET.has(char)) {
      hasSupportedSymbol = true;
      continue;
    }

    hasUnsupportedCharacter = true;
  }

  const charCount = Array.from(password).length;
  const hasValidLength = charCount >= PASSWORD_MIN_LENGTH && charCount <= PASSWORD_MAX_LENGTH;
  const hasValidSymbol = hasSupportedSymbol && !hasUnsupportedCharacter;

  return {
    hasValidLength,
    hasLetter,
    hasNumber,
    hasSupportedSymbol: hasValidSymbol,
    hasUnsupportedCharacter,
    isValid: hasValidLength && hasLetter && hasNumber && hasValidSymbol,
  };
}

export function passwordsMatch(password: string, confirmPassword: string) {
  return confirmPassword.length > 0 && password === confirmPassword;
}
