# RTX 4090 — Defect Evidence (for return/RMA)

## Summary
The card's PCIe interface does not respond to link training. It accepts 12V power and illuminates LEDs, but the GPU never enumerates on the PCIe bus in any slot on a known-good system. No BIOS, firmware, driver, or OS-level remedy exists for this failure mode — a PCIe endpoint must respond electrically before any software stack can interact with it.

## Test System (known-good)
- **Motherboard:** MSI MAG X570 Tomahawk WiFi, BIOS 1.J0 (latest)
- **CPU:** AMD Ryzen 5 5600X
- **RAM:** 1× 16 GB in DIMM A2 (correct slot)
- **PSU:** RM1000x, single 12VHPWR cable (native, not daisy-chained)
- **OS:** Ubuntu 24.04, kernel 6.17.0-22-generic
- **Control cards (both work in this system):** RTX 3050, RTX 3060, RTX 3080

## BIOS / firmware state verified correct
- UEFI boot mode (CSM disabled)
- Above 4G Decoding: enabled
- Resizable BAR: enabled (verified — 3060 maps 16 GB BAR at `0x7800000000`, 496 GB MMIO free above 4G)
- PCIe slot forced to Gen3
- Kernel args: `pci=realloc iommu=pt pcie_aspm=off pcie_port_pm=off`
- ACPI/DSDT: GPU bridge (`00:03.1`) has no power-gating or GPU-specific init logic
- IOMMU: passthrough, no device exclusions
- CMOS reset performed

## Tests performed with the RTX 4090

### 1. Slot 1 (CPU-direct x16, bridge `00:03.1`) — sole GPU
- Motherboard VGA debug LED solid on
- No POST, no GRUB, no OS
- Monitored for 5+ minutes: zero network response, zero signs of boot
- Same slot with 3060/3080: boots normally every time

### 2. Slot 2 (X570 chipset x16 physical / x4 electrical) — alongside 3060 in slot 1
- System booted normally on the 3060
- **4090 completely invisible:**
  - `lspci` across all 256 buses: not present
  - Manual config-space probe of every bus:dev:fn for vendor `10de`: not present
  - `echo 1 > /sys/bus/pci/rescan`: no new devices
  - Removal + rescan of chipset upstream bridge (`00:01.2`): no new devices
  - NVML: sees 1 GPU (the 3060)
  - I2C / SMBus scan: no GPU-side I2C adapters created (the 3060 created `i2c-3..8`; the 4090 created none)
- **Zero PCIe errors in `dmesg`** — the bus never saw the device to error on it
- PCIe link retrain and Secondary Bus Reset attempted on every downstream port: no link came up

### 3. Power delivery — confirmed good
- 12VHPWR connector: card's own power-indicator LED is **off** when cable is seated (= adequate power) and **on** when unplugged
- Cable: native RM1000x 12VHPWR (not daisy-chained, not adapter)
- Fully seated (audible click), inspected for bent/recessed pins

## Why "lights and soft-shutdown" are not evidence the GPU is alive
- The card's LEDs and power-indicator are driven by its **standalone power-management IC**, which runs off +12V independently of the GPU die.
- Soft-shutdown via the case power button is handled by the motherboard's ATX controller, not the GPU.
- None of these require the AD102 die or its PCIe PHY to be functional.

## Conclusion
Two different slots on two different root complexes (CPU and X570 chipset), zero PCIe link training in either, zero response to any form of reset or re-enumeration, and zero visibility on sideband interfaces (I2C/SMBus/NVML). The same motherboard and slots work correctly with three other NVIDIA cards.

This is a hardware defect in the 4090 — most likely a dead PCIe PHY on the AD102 die. The card is not fit for purpose and should be returned under warranty / DOA policy.
