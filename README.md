# bedrock-web-search-proxy

Drop-in [Perplexity Sonar API](https://docs.perplexity.ai/) replacement backed by [AWS Bedrock Nova grounding](https://aws.amazon.com/bedrock/). One URL change — real-time web search with citations, no Perplexity subscription needed.

Defaults to **Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`). Pass any US Nova CRIS profile ID directly to use a different model.

## Why reinvent the wheel?

Three reasons this exists:

**AWS credits you're not using.** AWS Community Builder accounts get $500/year in credits. Enterprise AWS accounts often have committed spend or promotional credits sitting idle. Bedrock calls count against that — Perplexity subscriptions don't.

**Company procurement policy.** Many teams have policies that make onboarding a new SaaS vendor (Brave Search, Perplexity, etc.) a multi-week process — security reviews, legal sign-off, purchase orders. Your AWS account is already approved.

**Credit card hygiene.** If you'd rather not hand your card to yet another AI startup, this keeps everything under one roof — your existing cloud provider.

If none of these apply to you, just use Perplexity. It's great. This is for everyone else.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and AWS credentials with Bedrock access (`bedrock:InvokeModel` + `bedrock:InvokeTool` on `us.amazon.nova-2-lite-v1:0`).

```bash
uvx --from git+https://github.com/gabrielkoo/bedrock-web-search-proxy bedrock-web-search-proxy
```

Or run directly without cloning:

```bash
uv run https://raw.githubusercontent.com/gabrielkoo/bedrock-web-search-proxy/master/main.py
```

Server starts on `http://localhost:7000`. Point your app's Perplexity base URL there.

## Configuration

All env vars are optional:

| Variable | Default | Description |
|---|---|---|
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region — Nova grounding available in `us-east-1`, `us-east-2`, `us-west-2` |
| `API_KEY` | _(none)_ | If set, all requests must include `Authorization: Bearer <key>` |
| `PORT` | `7000` | Port to listen on |
| `MAX_CONCURRENT` | `5` | Max concurrent Bedrock requests (429 when exceeded) |

Example with overrides:

```bash
AWS_DEFAULT_REGION=us-east-1 API_KEY=secret PORT=8000 uv run https://raw.githubusercontent.com/gabrielkoo/bedrock-web-search-proxy/master/main.py
```

## Model Routing

All standard Perplexity model names (`sonar`, `sonar-pro`, etc.) and Nova aliases (`nova-2-lite-grounding`, `nova-premier-grounding`, etc.) default to `us.amazon.nova-2-lite-v1:0`.

To use a specific model, pass the CRIS profile ID directly:
- `us.amazon.nova-premier-v1:0` — used as-is
- `amazon.nova-premier-v1:0` — `us.` prefix added automatically

All 5 US Nova CRIS profiles support web grounding: `nova-premier`, `nova-pro`, `nova-lite`, `nova-micro`, `nova-2-lite`.

## Connecting Your App

For **OpenClaw** (`~/.openclaw/openclaw.json`):

```json
{
  "tools": {
    "web": {
      "search": {
        "provider": "perplexity",
        "perplexity": {
          "baseUrl": "http://localhost:7000/v1",
          "apiKey": "nova-grounding",
          "model": "sonar"
        }
      }
    }
  }
}
```

> ⚠️ `apiKey` must **not** be a real `pplx-` key — that prefix causes OpenClaw to override `baseUrl` back to Perplexity's servers.

For any other app (Open WebUI, LibreChat, Cursor, Continue.dev, AnythingLLM, LiteLLM): set the Perplexity base URL to `http://localhost:7000/v1`. All standard Perplexity model names are accepted and routed to Nova 2 Lite by default.

## Lambda Function URL (Serverless)

Deploy as a Lambda Function URL instead of running locally — no server, pay-per-use:

```bash
git clone https://github.com/gabrielkoo/bedrock-web-search-proxy
cd bedrock-web-search-proxy
sam build
sam deploy --guided  # set ApiKey when prompted
```

No Docker needed — `sam build` pulls pre-built manylinux wheels for your architecture (arm64 or x86_64). Uses Lambda Web Adapter (LWA) for streaming support.

**Why Lambda Function URL?**

- **Native streaming** — `InvokeMode: RESPONSE_STREAM` handles SSE natively, no extra config
- **No Docker** — zip-based deploy; `sam build` handles dependencies without a container registry
- **No API Gateway or ALB** — Function URLs are a direct HTTPS endpoint, saving ~$3.50+/month minimum
- **Pay-as-you-go** — cents/month at typical usage, covered by AWS Free Tier or Community Builder credits

**Running OpenClaw on your own instance?** If you're on a Raspberry Pi, VPS, or home server and don't mind a small memory footprint (~50 MB), running this adapter as a `systemd` user service is simpler — no AWS deployment needed. Use the `uvx` or `uv run` method above and wrap it in a service unit.

## Proof It's Grounded

```bash
curl -s http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sonar","messages":[{"role":"user","content":"What is the Bitcoin price right now?"}]}'
```

Response includes `citations[]` with real URLs — e.g. a Binance price page and a news article with today's date in the slug. Not hallucinated.

## Caveats

- Streaming responses don't include `citations[]` — Nova limitation. Non-streaming works fine.
- Nova grounding is only available in US regions: `us-east-1`, `us-east-2`, `us-west-2`.

## License

MIT
