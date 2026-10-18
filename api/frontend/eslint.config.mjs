import nextConfig from "eslint-config-next";

const eslintConfig = [
  {
    // `public/` is shipped verbatim and is not source. Everything in it is
    // generated: `sw.js` is written by serwist on every build, and
    // `site-assets/reference/support.js` is a bundle built from `dc-runtime/src`
    // that says "do not edit" in its first line. Linting them reported two
    // errors that cannot be fixed in place — an edit is overwritten by the next
    // build — and both were false positives against generated code anyway: a
    // deliberate `ReactDOM.render` fallback guarded by a `createRoot` check, and
    // a `const module = { exports: {} }` local implementing a CommonJS shim,
    // which is not the webpack `module` global that rule is about.
    ignores: ["public/**"],
  },
  ...nextConfig,
  {
    rules: {
      // Downgrade new React 19 strict rules to warnings while we migrate
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/static-components": "warn",
    },
  },
];

export default eslintConfig;
