# One deployment per institution

Each institution gets its own Qdrant instance and its own pipeline instance. There is no tenant
identifier anywhere in the schema, and no query carries a tenant filter.

Two institutions are in scope and they are public bodies with unrelated corpora, so isolation is
worth more than shared infrastructure. Making leakage structurally impossible is stronger than
relying on every query remembering a filter, and per-institution deployments can diverge later, for
example one moving to Montenegrin documents before the other.

## Consequences

Operating N institutions means operating N deployments. At two customers this is trivial. Somewhere
around ten it becomes a burden, and the decision should be revisited then rather than assumed to
still hold.

Retrofitting shared tenancy later means introducing a tenant field, backfilling it, adding a
`is_tenant: true` payload index, and auditing every query path. That is the cost of being wrong.
