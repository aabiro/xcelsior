# Xcelsior TypeScript SDK

Typed JavaScript and TypeScript client for the Xcelsior Compute API.

## Requirements

- Node.js 18 or newer
- An Xcelsior API token for authenticated endpoints

## Install

```sh
npm install @xcelsior-gpu/sdk
```

## Create a client

```ts
import {
  XcelsiorApiClient,
  XcelsiorApiEnvironment,
} from '@xcelsior-gpu/sdk'

const token = process.env.XCELSIOR_API_TOKEN
if (!token) throw new Error('Set XCELSIOR_API_TOKEN before making API calls')

const client = new XcelsiorApiClient({
  token,
  environment: XcelsiorApiEnvironment.Production,
})

const instances = await client.instances.list()
console.log(instances.body)
```

The production environment is `https://xcelsior.ca`. Use `baseUrl` to connect
to a compatible self-hosted API:

```ts
const client = new XcelsiorApiClient({
  token,
  baseUrl: 'https://xcelsior.example.com',
  timeoutInSeconds: 30,
  maxRetries: 2,
})
```

## API clients

The root client exposes `artifacts`, `auth`, `billing`, `inference`,
`instances`, `marketplace`, `marketplaceV2`, `serverless`, and `volumes`
resources. Each resource has typed request and response definitions. The client
also supports custom fetch implementations, request-level timeouts and retries,
additional headers, and passthrough requests through `client.fetch()`.

See the [API reference](https://docs.xcelsior.ca/api-reference) for endpoint
details and the [SDK source](https://github.com/aabiro/xcelsior/tree/main/api/frontend/sdk)
for generated type definitions.

## License

MIT
