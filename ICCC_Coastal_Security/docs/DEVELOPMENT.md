# Development and documentation process

Develop one reviewable vertical slice at a time. Preserve application capabilities and history, use synthetic fixtures, and attach validation evidence to the same change that introduces behavior.

## Change sequence

1. Choose an item from [ROADMAP.md](ROADMAP.md). Describe the user action, current result, expected result and acceptance cases.
2. Record the affected project and revision: research repository, ICCC application or separate Lovable coastal dashboard. Changes to one do not update the others automatically.
3. Implement the smallest complete slice. Retain existing authorized workflows. For security changes, include both successful authorized and denied unauthorized cases.
4. Run the relevant checks in [VALIDATION.md](VALIDATION.md). Record command, environment, revision, result and unresolved failures. A skipped or unavailable test stays explicitly unverified.
5. Update the capability inventory, affected architecture/workflow documents, installation instructions and backlog status in the same PR. Regenerate API/RBAC/schema references when their sources change.
6. Review the actual diff and CI for that revision. Separate source review, test success, preview verification and production validation.
7. Release only after the applicable charter gates pass. Tag the exact tested commit, publish change notes, dependency/attribution inventory, known limitations and rollback instructions. Publishing or merging is a separate decision from building a preview.

## Documentation ownership

The contributor owns the documentation and tests for their change; a reviewer verifies the evidence. Individual operational/data owners must be designated before a pilot, rather than assumed from a software role name.

| Change | Documents to update |
| --- | --- |
| Route, field, role or jurisdiction | API/RBAC/schema references, capability inventory, negative tests |
| Incident/tasking behavior | Incident and command workflows, validation scenario, acceptance tests |
| Data source or model | Provenance/license record, dataset/model card, architecture, metrics and limitations |
| Deployment/dependencies | Installation, lockfiles, CI, security review and release notes |
| Lovable interface | Separate project's feature inventory, preview evidence, integration boundary here if changed |

From the ICCC directory, `python start_scripts/gen_docs.py` regenerates the API, RBAC and table reference files. Run it with development dependencies installed and review the diff; generated permission tables describe configured policy, not proof that every handler enforces it.

## Evidence record template

```text
Change / backlog ID:
Project and commit:
Implemented behavior:
Preserved capabilities:
Data provenance and authorization:
Commands / environment / results:
CI or artifact links:
Known failures and limitations:
Reviewer / acceptance date:
Rollback revision or procedure:
```

Do not put credentials, personal information, real incident data or private planning chats into commits, logs or screenshots. Maintain separate reviewed demo fixtures. Demonstration labels and human review requirements remain visible as the implementation evolves.
