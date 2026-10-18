// check-labels.ts — Plain names for check results.
//
// Check results carry the names the logic, the server and the tests match on
// ("FP16 Matmul", "runc", "Admission Boundary"). People saw those verbatim.
// This maps them to what they mean at the two places a name is drawn: the
// results list and the live progress log. Unknown names pass through.

const PLAIN_CHECK_NAMES: Record<string, string> = {
    "API Connection": "Connection to Xcelsior",
    "Admission": "Approval",
    "Admission Boundary": "Approval",
    "CUDA Readiness": "GPU software (CUDA)",
    "FP16 Matmul": "GPU speed",
    "GPU Identity": "GPU model",
    "Host Registration": "Marketplace listing",
    "Jitter": "Connection stability",
    "Memory Integrity": "GPU memory",
    "Mesh Network": "Private network",
    "OAuth client": "App key",
    "PCIe Bandwidth": "Data transfer speed",
    "Packet Loss": "Dropped data",
    "Project root": "Project folder",
    "Runtime": "Container runtime",
    "SDK connection": "SDK test call",
    "Security Posture": "Security settings",
    "Server Verification": "Xcelsior's checks",
    "Thermal Stability": "Temperature under load",
    "Throughput": "Speed",
    "Version Compatibility": "Versions",
    "Worker Agent": "Xcelsior agent",
    "runc": "Container runtime (runc)",
};

export function displayCheckName(name: string): string {
    return PLAIN_CHECK_NAMES[name] ?? name;
}
