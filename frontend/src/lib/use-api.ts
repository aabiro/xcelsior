"use client";

import { useMemo } from "react";
import * as api from "@/lib/api";

/**
 * Hook returning API functions. Auth is handled via httpOnly cookies —
 * no token binding needed. The returned object is memoized so it can
 * safely appear in useCallback / useEffect dependency arrays.
 */
export function useApi() {
  return useMemo(() => ({
    fetchHosts: api.fetchHosts,
    registerHost: api.registerHost,
    registerHostWeb: api.registerHostWeb,
    fetchInstances: api.fetchInstances,
    submitInstance: api.submitInstance,
    cancelInstance: api.cancelInstance,
    requeueInstance: api.requeueInstance,
    fetchBilling: api.fetchBilling,
    fetchWallet: api.fetchWallet,
    fetchMarketplace: api.fetchMarketplace,
    searchMarketplace: api.searchMarketplace,
    fetchTelemetry: api.fetchTelemetry,
    fetchPricingReference: api.fetchPricingReference,
    fetchReservedPlans: api.fetchReservedPlans,
    fetchSpotPrices: api.fetchSpotPrices,
    fetchLeaderboard: api.fetchLeaderboard,
    fetchAnalytics: api.fetchAnalytics,
    fetchEnhancedAnalytics: api.fetchEnhancedAnalytics,
    fetchWalletHistory: api.fetchWalletHistory,
    fetchUsageSummary: api.fetchUsageSummary,
    fetchProviderEarnings: api.fetchProviderEarnings,
    fetchSpotPricesV2: api.fetchSpotPricesV2,
    fetchSpotHistory: api.fetchSpotHistory,
    fetchSlaHostsSummary: api.fetchSlaHostsSummary,
    fetchMarketplaceStatsV2: api.fetchMarketplaceStatsV2,
    fetchBurstStatus: api.fetchBurstStatus,
  }), []);
}
