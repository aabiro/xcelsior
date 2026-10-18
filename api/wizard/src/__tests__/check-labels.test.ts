// Check names are internal ("FP16 Matmul", "runc"); people see plain ones.

import { describe, it, expect } from "vitest";
import { displayCheckName } from "../check-labels.js";

describe("displayCheckName", () => {
    it("shows what a technical check measures", () => {
        expect(displayCheckName("FP16 Matmul")).toBe("GPU speed");
        expect(displayCheckName("Admission Boundary")).toBe("Approval");
        expect(displayCheckName("runc")).toBe("Container runtime (runc)");
    });

    it("passes through names that are already plain", () => {
        expect(displayCheckName("Docker")).toBe("Docker");
        expect(displayCheckName("Wallet")).toBe("Wallet");
    });
});
