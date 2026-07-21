import { fauxAssistantMessage, fauxProvider } from "@earendil-works/pi-ai";

export default function (pi) {
  const provider = fauxProvider({
    provider: "ark-slow-probe",
    api: "ark-probe-api",
    tokensPerSecond: 1,
    models: [{
      id: "faux-slow",
      name: "ARK Slow Probe",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 4096,
    }],
  });
  provider.setResponses([
    fauxAssistantMessage("ARK_PI_SLOW_" + "0123456789".repeat(100)),
  ]);
  const model = provider.models[0];
  pi.registerProvider("ark-slow-probe", {
    name: "ARK Slow Probe",
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
}

