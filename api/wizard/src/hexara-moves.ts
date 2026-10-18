// hexara-moves.ts — Frame plan for new Hexara moves
//
// PNG source: sprites/wizard/generate_wizard_sprites.py --only peek,type,nod
// Sixel regen: npm run generate-frames

import {
    PEEK_FRAMES,
    TYPE_FRAMES,
    NOD_FRAMES,
} from "../sprites/wizard/wizard-frames.js";

export { PEEK_FRAMES, TYPE_FRAMES, NOD_FRAMES };

/** Artist-facing spec — drop PNGs named wizard-<move>-N.png to replace placeholders. */
export const HEXARA_MOVE_FRAME_PLAN = {
    peek: {
        purpose: "Curious inspection — scanning project tree, reading package.json, browsing marketplace.",
        frameCount: 6,
        filePattern: "wizard-peek-{1..6}.png",
        keyPoses: [
            "1 Neutral stand, eyes forward",
            "2 Head tilt left, one eye slightly larger",
            "3 Lean forward, hat brim dips, wand lowered",
            "4 Peek past hat brim (squint / look up)",
            "5 Return upright, small sparkle near eye",
            "6 Settle to idle (match wizard-idle-1 feet on stage)",
        ],
        wizardSteps: ["sdk-detect", "sdk-install", "gpu-detect", "browse-gpus", "gpu-preference", "showAiPrompt"],
    },
    type: {
        purpose: "Writing config — .env.local, OAuth client, SSH keys, worker install commands.",
        frameCount: 6,
        filePattern: "wizard-type-{1..6}.png",
        keyPoses: [
            "1 Stand, wand becomes stylus/keyboard pointer",
            "2 Left hand (or robe fold) on imaginary keyboard",
            "3 Tap key — small yellow pixel flash",
            "4 Tap key — flash on other side",
            "5 Quick double-tap, brief code glyph (xoa_) above hands",
            "6 Hands down, satisfied nod hint (transition to nod-1)",
        ],
        wizardSteps: ["sdk-credentials", "ssh-key-setup", "worker-install", "confirm-setup"],
    },
    nod: {
        purpose: "Affirmative ack — confirm dialogs, pricing locked, package found, ready to ship.",
        frameCount: 4,
        filePattern: "wizard-nod-{1..4}.png",
        keyPoses: [
            "1 Neutral (reuse idle pose)",
            "2 Chin down ~2px — yes bob",
            "3 Chin up past neutral — slight overshoot",
            "4 Return to neutral with tiny smile line / closed happy eyes",
        ],
        wizardSteps: ["sdk-snippet", "sdk-install", "pricing", "image-pick", "spot-enabled", "provider-summary"],
    },
} as const;