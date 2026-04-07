import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ── Fun random instance name generator (adjective-noun) ─────────── */

const ADJECTIVES = [
  "wobbly", "chunky", "sneaky", "peppy", "grumpy", "sparkly", "fluffy",
  "zippy", "squishy", "dizzy", "bouncy", "toasty", "wiggly", "fuzzy",
  "jolly", "snappy", "breezy", "cheeky", "quirky", "plucky", "zappy",
  "perky", "spunky", "nifty", "sassy", "wacky", "frisky", "goofy",
  "turbo", "cosmic", "mighty", "sleepy", "groovy", "dapper", "zesty",
];

const NOUNS = [
  "panda", "otter", "moose", "walrus", "narwhal", "llama", "penguin",
  "quokka", "axolotl", "capybara", "gecko", "badger", "wombat", "sloth",
  "toucan", "bison", "falcon", "corgi", "puffin", "mantis", "goblin",
  "yeti", "kraken", "phoenix", "dragon", "taco", "waffle", "noodle",
  "pickle", "muffin", "pretzel", "donut", "nugget", "dumpling", "turnip",
];

export function generateFunName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  return `${adj}-${noun}`;
}
