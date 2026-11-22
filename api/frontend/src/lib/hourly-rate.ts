/**
 * An hourly GPU rate as a renter reads it: cents under $1 keep a third decimal
 * when it carries one ($0.044, $0.096), so a cheap card's rate is not rounded
 * away; $1 and up show two.
 */
export function formatHourlyRate(value: number): string {
  if (value >= 1) return `$${value.toFixed(2)}`;
  const text = value.toFixed(3);
  return `$${text.endsWith("0") ? text.slice(0, -1) : text}`;
}
