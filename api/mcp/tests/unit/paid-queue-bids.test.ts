import { describe, expect, it } from "vitest";
import { z } from "zod";
import { registerComputeTools } from "../../src/tools/compute.js";
import { registerDiscoveryTools } from "../../src/tools/discovery.js";
import type { AuthUser } from "../../src/auth/bearer.js";

function harness() {
  type Handler = (args: Record<string, unknown>) => Promise<unknown>;
  const tools = new Map<string, { schema: z.ZodObject<z.ZodRawShape>; handler: Handler }>();
  const calls: unknown[] = [];
  const server = {
    registerTool(name: string, config: { inputSchema: z.ZodObject<z.ZodRawShape> }, handler: Handler) {
      tools.set(name, {schema: config.inputSchema,handler});
    },
  };
  const client = {
    async get(path: string, query: unknown) { calls.push({path,query}); return {ok:true}; },
    async post(path: string, body: unknown) { calls.push({path,body}); return {ok:true,plan_id:"plan-paid"}; },
  };
  const user = {scopes:["gpu:read","instances:write"]} as AuthUser;
  registerComputeTools(server as never,client as never,user);
  registerDiscoveryTools(server as never,client as never,user);
  const call = async (name: string, input: Record<string, unknown>) => {
    const tool = tools.get(name)!;
    return tool.handler(tool.schema.parse(input));
  };
  return {calls,call};
}

describe("paid queue bid contracts",()=>{
  it("lists exact Stripe prices for the requested GPU and pricing mode",async()=>{
    const h=harness();
    await h.call("list_tiers",{gpu_model:"RTX 4090",pricing_mode:"spot"});
    expect(h.calls).toEqual([{path:"/api/pricing/service-tiers",query:{gpu_model:"RTX 4090",mode:"spot"}}]);
  });

  it("forwards both paid add-on identities into a launch preview",async()=>{
    const h=harness();
    await h.call("create_instance",{name:"paid-bid",gpu_model:"RTX 4090",tier:"premium",service_tier_price_id:"price_premium",exclusive_gpu:true,gpu_exclusive_price_id:"price_exclusive",queue_bid_units:20,queue_bid_price_id:"price_bid",priority:999});
    expect(h.calls).toEqual([expect.objectContaining({path:"/api/v1/launch-plans",body:expect.objectContaining({tier:"premium",queue_bid_units:20,queue_bid_price_id:"price_bid",gpu_exclusive_price_id:"price_exclusive"})})]);
    expect((h.calls[0] as {body:Record<string,unknown>}).body).not.toHaveProperty("priority");
  });

  it("refuses invalid bid quantities before making an API call",async()=>{
    const h=harness();
    for(const queue_bid_units of [-1,0.2,1000001]){
      await expect(h.call("create_instance",{name:"invalid-bid",queue_bid_units})).rejects.toThrow();
    }
    expect(h.calls).toEqual([]);
  });
});
