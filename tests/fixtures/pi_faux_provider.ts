import {
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
} from "@earendil-works/pi-ai";

export default function (pi) {
  const provider = fauxProvider({
    provider: "ark-probe",
    api: "ark-probe-api",
    tokensPerSecond: 2000,
    models: [{
      id: "faux-1",
      name: "ARK Probe Faux",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 4096,
    }],
  });
  provider.setResponses([
    fauxAssistantMessage(
      fauxToolCall("write", { path: "pi_provider_probe.txt", content: "ARK_PI_PROVIDER_OK\n" }, { id: "write-1" }),
      { stopReason: "toolUse" },
    ),
    fauxAssistantMessage("ARK_PI_PROVIDER_DONE"),
  ]);
  const model = provider.models[0];
  pi.registerProvider("ark-probe", {
    name: "ARK Probe Faux",
    baseUrl: model.baseUrl,
    apiKey: "probe-no-network",
    api: provider.api,
    streamSimple: provider.provider.streamSimple,
    models: [{
      id: model.id,
      name: model.name,
      reasoning: model.reasoning,
      input: model.input,
      cost: model.cost,
      contextWindow: model.contextWindow,
      maxTokens: model.maxTokens,
    }],
  });
  pi.on("session_before_compact", async (event) => ({
    compaction: {
      summary: "ARK_PI_COMPACTION_SUMMARY",
      firstKeptEntryId: event.preparation.firstKeptEntryId,
      tokensBefore: event.preparation.tokensBefore,
      details: { arkProviderTest: true },
    },
  }));
}
