import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key} ${Object.values(vars).join(" ")}` : key,
  }),
}));
const toastError = vi.fn();
const toastSuccess = vi.fn();
vi.mock("sonner", () => ({
  toast: { success: (...a: unknown[]) => toastSuccess(...a), error: (...a: unknown[]) => toastError(...a) },
}));
const updateProfile = vi.fn();
const uploadBusinessLogo = vi.fn();
const deleteBusinessLogo = vi.fn();
vi.mock("@/lib/api", () => ({
  updateProfile: (...a: unknown[]) => updateProfile(...a),
  uploadBusinessLogo: (...a: unknown[]) => uploadBusinessLogo(...a),
  deleteBusinessLogo: (...a: unknown[]) => deleteBusinessLogo(...a),
}));

import type { User } from "@/lib/auth";
import {
  businessName,
  monogram,
  monogramGradient,
  websiteIsValid,
  websiteLabel,
} from "@/lib/business-profile";
import { BusinessLogo } from "@/components/user/business-logo";
import { BusinessMenuCard } from "@/components/user/business-identity";
import { BusinessProfileSection, BusinessProfileTeaser } from "@/components/settings/business-profile-section";

const business = (over: Partial<User> = {}): User => ({
  user_id: "u-1",
  email: "dana@example.test",
  name: "Dana Founder",
  role: "submitter",
  account_type: "business",
  business_name: "North Labs Inc.",
  business_display_name: "",
  business_website: "",
  business_tagline: "",
  business_logo_url: null,
  business_logo_tile: "none",
  ...over,
});

describe("business profile helpers", () => {
  it("accepts the addresses the API accepts and refuses the rest", () => {
    for (const ok of ["", "example.com", "https://Example.COM/", "www.example.co.uk/about", "bücher.de"]) {
      expect(websiteIsValid(ok), ok).toBe(true);
    }
    for (const bad of [
      "javascript:alert(1)",
      "ftp://example.com",
      "https://user:secret@example.com",
      "https://localhost",
      "https://10.0.0.1",
      "exa mple.com",
      "https://example",
    ]) {
      expect(websiteIsValid(bad), bad).toBe(false);
    }
  });

  it("names the business by its display name, else its legal name", () => {
    expect(businessName({ business_display_name: " North ", business_name: "North Labs Inc." })).toBe("North");
    expect(businessName({ business_display_name: "", business_name: "North Labs Inc." })).toBe("North Labs Inc.");
  });

  it("takes initials by letter and keeps a colour per name", () => {
    expect(monogram("North Labs")).toBe("NL");
    expect(monogram("North Labs", 1)).toBe("N");
    expect(monogram("élan studio")).toBe("ÉS");
    expect(monogram("  (Acme)  ")).toBe("A");
    expect(monogramGradient("North Labs")).toBe(monogramGradient("north labs"));
  });

  it("shows a website without its scheme or www", () => {
    expect(websiteLabel("https://www.northlabs.example/about/")).toBe("northlabs.example/about");
    expect(websiteLabel("")).toBe("");
  });
});

describe("BusinessLogo", () => {
  it("draws initials on a gradient when there is no logo", () => {
    const { container } = render(<BusinessLogo business={business({ business_display_name: "North Labs" })} />);
    const tile = container.querySelector(".business-logo")!;
    expect(tile.getAttribute("data-tile")).toBe("monogram");
    expect(tile.textContent).toBe("NL");
    expect((tile as HTMLElement).style.backgroundImage).toContain("linear-gradient");
  });

  it("puts a logo on the tile the API chose, and falls back if it will not load", () => {
    const { container } = render(
      <BusinessLogo business={business({ business_logo_url: "/logo?v=1", business_logo_tile: "light" })} />,
    );
    const tile = container.querySelector(".business-logo")!;
    expect(tile.getAttribute("data-tile")).toBe("light");
    fireEvent.error(container.querySelector("img")!);
    expect(tile.getAttribute("data-tile")).toBe("monogram");
    expect(tile.textContent).toBe("NL");
  });
});

describe("BusinessMenuCard", () => {
  it("links the website safely, in a new tab", () => {
    render(
      <BusinessMenuCard
        business={business({ business_website: "https://www.northlabs.example" })}
        badgeLabel="Business"
        visitLabel={(site) => `Visit ${site}`}
      />,
    );
    const link = screen.getByRole("link", { name: "Visit northlabs.example" });
    expect(link.getAttribute("href")).toBe("https://www.northlabs.example");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
  });
});

describe("BusinessProfileSection", () => {
  const OriginalImage = globalThis.Image;

  beforeEach(() => {
    updateProfile.mockReset();
    uploadBusinessLogo.mockReset();
    deleteBusinessLogo.mockReset();
    toastError.mockReset();
    toastSuccess.mockReset();
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
    // jsdom loads no images; let the preload finish at once.
    globalThis.Image = class {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      set src(_value: string) {
        setTimeout(() => this.onload?.(), 0);
      }
    } as unknown as typeof Image;
  });
  afterEach(() => {
    globalThis.Image = OriginalImage;
  });

  it("saves only when something changed, and shows what the API stored", async () => {
    updateProfile.mockResolvedValue({
      ok: true,
      business_display_name: "North",
      business_website: "https://northlabs.example",
      business_tagline: "",
    });
    // As in the app: saving refreshes the account, which re-renders the section.
    const stored = business({ business_display_name: "North", business_website: "https://northlabs.example" });
    const onUpdated = vi.fn(async () => {
      view.rerender(<BusinessProfileSection user={stored} onUpdated={onUpdated} />);
    });
    const view = render(<BusinessProfileSection user={business()} onUpdated={onUpdated} />);

    const save = screen.getByRole("button", { name: /dash.business.save/ });
    expect(save).toHaveProperty("disabled", true);

    fireEvent.change(screen.getByLabelText(/dash.business.display_name/), { target: { value: "  North " } });
    fireEvent.change(screen.getByLabelText(/dash.business.website/), { target: { value: "northlabs.example" } });
    expect(save).toHaveProperty("disabled", false);
    fireEvent.click(save);

    await waitFor(() =>
      expect(updateProfile).toHaveBeenCalledWith({
        business_display_name: "North",
        business_website: "northlabs.example",
        business_tagline: "",
      }),
    );
    await waitFor(() => expect(onUpdated).toHaveBeenCalled());
    expect((screen.getByLabelText(/dash.business.website/) as HTMLInputElement).value).toBe(
      "https://northlabs.example",
    );
    expect(await screen.findByRole("status")).toHaveProperty("textContent", "dash.business.saved");
    expect(save).toHaveProperty("disabled", true);
  });

  it("explains a bad website once the field is left, and will not save it", () => {
    render(<BusinessProfileSection user={business()} onUpdated={vi.fn()} />);
    const website = screen.getByLabelText(/dash.business.website/);
    fireEvent.change(website, { target: { value: "javascript:alert(1)" } });
    expect(screen.queryByText("dash.business.website_invalid")).toBeNull();
    fireEvent.blur(website);
    expect(screen.getByText("dash.business.website_invalid")).toBeTruthy();
    expect(website.getAttribute("aria-invalid")).toBe("true");
    expect(screen.getByRole("button", { name: /dash.business.save/ })).toHaveProperty("disabled", true);
  });

  it("does not end the legal-name hint with a second full stop", () => {
    render(<BusinessProfileSection user={business()} onUpdated={vi.fn()} />);
    expect(screen.getByText("dash.business.display_name_hint North Labs Inc.")).toBeTruthy();
  });

  it("refuses a file that is not a logo before uploading anything", async () => {
    const { container } = render(<BusinessProfileSection user={business()} onUpdated={vi.fn()} />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["%PDF"], "logo.pdf", { type: "application/pdf" })] } });
    expect(toastError).toHaveBeenCalledWith("dash.business.logo_type");
    const big = new File([new Uint8Array(2 * 1024 * 1024 + 1)], "big.png", { type: "image/png" });
    fireEvent.change(input, { target: { files: [big] } });
    expect(toastError).toHaveBeenCalledWith("dash.business.logo_too_big");
    expect(uploadBusinessLogo).not.toHaveBeenCalled();
  });

  it("uploads a dropped logo and refreshes the account", async () => {
    uploadBusinessLogo.mockResolvedValue({
      ok: true,
      business_logo_url: "/api/auth/me/business-logo?v=2",
      business_logo_tile: "light",
    });
    const onUpdated = vi.fn().mockResolvedValue(undefined);
    render(<BusinessProfileSection user={business()} onUpdated={onUpdated} />);
    const file = new File(["png"], "logo.png", { type: "image/png" });
    const zone = screen.getByText(/dash.business.logo_drop/).closest("div")!;
    await act(async () => {
      fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    });
    await waitFor(() => expect(uploadBusinessLogo).toHaveBeenCalledWith(file));
    await waitFor(() => expect(onUpdated).toHaveBeenCalled());
    expect(toastSuccess).toHaveBeenCalledWith("dash.business.logo_uploaded");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview");
  });

  it("removes the logo", async () => {
    deleteBusinessLogo.mockResolvedValue({ ok: true, removed: true });
    const onUpdated = vi.fn().mockResolvedValue(undefined);
    render(
      <BusinessProfileSection
        user={business({ business_logo_url: "/api/auth/me/business-logo?v=1", business_logo_tile: "dark" })}
        onUpdated={onUpdated}
      />,
    );
    expect(screen.getByText("dash.business.tile_dark")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /dash.business.logo_remove/ }));
    await waitFor(() => expect(deleteBusinessLogo).toHaveBeenCalled());
    await waitFor(() => expect(onUpdated).toHaveBeenCalled());
  });
});

describe("BusinessProfileTeaser", () => {
  it("takes a personal account to the account type field", () => {
    document.body.insertAdjacentHTML("beforeend", '<select data-field="account-type"></select>');
    const select = document.querySelector<HTMLSelectElement>('[data-field="account-type"]')!;
    select.scrollIntoView = vi.fn();
    render(<BusinessProfileTeaser />);
    fireEvent.click(screen.getByRole("button", { name: "dash.business.locked_action" }));
    expect(select.scrollIntoView).toHaveBeenCalled();
    expect(document.activeElement).toBe(select);
    select.remove();
  });
});
