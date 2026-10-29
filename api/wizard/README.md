# Xcelsior setup wizard

An interactive terminal guide for getting started with Xcelsior. It helps GPU
renters launch compute, hardware providers onboard a machine, and application
developers connect a Node.js project to the Xcelsior SDK.

## Requirements

- Node.js 18 or newer
- An interactive terminal
- For GPU provider setup: Docker and compatible GPU hardware; the wizard checks
  the local environment before continuing

## Run it

Start the latest published wizard without installing it globally:

```bash
npx --yes @xcelsior-gpu/wizard
```

Or install the `xcelsior-wizard` command globally:

```bash
npm install --global @xcelsior-gpu/wizard
xcelsior-wizard
```

Choose a workflow when prompted:

- **Rent GPUs** to browse capacity, configure a job, and launch it.
- **Provide GPUs** to check a machine and work through provider onboarding.
- **Both** to set up renting and providing from the same account.
- **Build with the SDK** to check a Node.js project and configure its Xcelsior
  integration. Run this option from the project directory containing
  `package.json`.

Sign in through the browser-based device flow. The wizard also offers a manual
token entry path if device authorization is unavailable. Review any payment,
compute launch, or provider setup details shown in the terminal before
confirming them.

## Credentials and local files

The wizard stores its local state under `~/.xcelsior` by default. Access tokens
are written to `token.json` with owner-only permissions; resumable wizard state
omits credential values. Set `XCELSIOR_CONFIG_DIR` to use a different
configuration directory.

To use a compatible self-hosted API, set `XCELSIOR_API_URL` before starting the
wizard. Leave it unset for the production service.

## Help

- [Xcelsior documentation](https://docs.xcelsior.ca)
- [Report an issue](https://github.com/aabiro/xcelsior/issues)
- [Source repository](https://github.com/aabiro/xcelsior/tree/main/api/wizard)

This package is distributed under the [MIT License](LICENSE).
