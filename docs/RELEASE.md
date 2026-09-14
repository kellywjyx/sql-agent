# Release checklist

1. Run independent Windows installation, offline tests, and pip check.
2. Run local model/GPU evaluation sequentially; preserve immutable run IDs and source hashes.
3. Review exported evidence for private data, source/asset licenses and unsupported claims.
4. Obtain the owner's GitHub identity and explicit publication approval. Do not invent repository URLs.
5. Publish the llm-evals wheel first and verify its SHA-256 against evals-dependency.json.
6. Configure consumer CI variable LLM_EVALS_WHEEL_URL with that immutable release artifact; the repository pins the expected filename and checksum.
7. Publish project source and approved compact evidence; exclude weights, databases, uploads, adapters, environments and indexes.

Hosted CI does not establish model quality. No untrusted pull-request job may run on the owner's GPU machine. Current workflows are prepared locally, not verified on GitHub.
