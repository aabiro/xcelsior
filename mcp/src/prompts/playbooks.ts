import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

export function registerPlaybooks(server: McpServer): void {
  server.registerPrompt(
    "cheapest-gpu-now",
    {
      title: "Find cheapest available GPU",
      description: "Playbook: discover spot/on-demand options under a budget",
      argsSchema: {
        max_hourly_cad: z.number().positive().optional().describe("Max CAD per hour"),
        gpu_model: z.string().optional().describe("Optional GPU model filter"),
      },
    },
    async ({ max_hourly_cad, gpu_model }) => {
      const budget = max_hourly_cad ? ` under $${max_hourly_cad.toFixed(2)} CAD/hr` : "";
      const model = gpu_model ? ` for ${gpu_model}` : "";
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: [
                `Find the cheapest GPU capacity available right now${model}${budget}.`,
                "1. Call list_available_gpus and get_spot_prices.",
                "2. Call should_i_run_this before any launch.",
                "3. If approved, use schedule_under_budget or create_instance with confirm:false first.",
              ].join("\n"),
            },
          },
        ],
      };
    },
  );

  server.registerPrompt(
    "ca-fine-tune",
    {
      title: "Canadian fine-tuning job",
      description: "Playbook: launch a CA-resident training instance with guardrails",
      argsSchema: {
        gpu_model: z.string().default("RTX 4090"),
        git_repo: z.string().optional(),
      },
    },
    async ({ gpu_model, git_repo }) => {
      const repo = git_repo ? ` Clone ${git_repo}.` : "";
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: [
                `Launch a Canadian-resident fine-tuning instance on ${gpu_model}.${repo}`,
                "1. Call should_i_run_this with require_canada:true.",
                "2. Prefer ca-east region when creating the instance.",
                "3. Use run_training_job with confirm:false for preview, then confirm:true to launch.",
                "4. Use watch_instance to monitor GPU utilization.",
              ].join("\n"),
            },
          },
        ],
      };
    },
  );

  server.registerPrompt(
    "serverless-inference",
    {
      title: "Run serverless inference",
      description: "Playbook: list endpoints, run job, poll status",
      argsSchema: {
        endpoint_id: z.string().optional(),
        prompt: z.string().optional(),
      },
    },
    async ({ endpoint_id, prompt }) => {
      const ep = endpoint_id ? ` endpoint ${endpoint_id}` : "";
      const userPrompt = prompt || "Summarize this workload in one paragraph.";
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: [
                `Run serverless inference${ep}.`,
                `User prompt: ${userPrompt}`,
                "1. list_serverless_endpoints if endpoint_id is unknown.",
                "2. run_serverless_job with input.messages or input.prompt.",
                "3. Poll get_serverless_job_status until completed or failed.",
              ].join("\n"),
            },
          },
        ],
      };
    },
  );
}