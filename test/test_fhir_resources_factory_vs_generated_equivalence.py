"""
Equivalence guard between factory-built and generated (checked-in) models.

The dynamic factory (``FHIRModelFactory``) and the checked-in generated modules
(``get_fhir_type`` / ``get_fhir_type_by_url``) must produce models that agree on
Pydantic field constraint metadata (``min_length``/``max_length``/``ge``/``le``)
for the same StructureDefinition.  This is the regression guard for
https://github.com/luisfabib/fhircraft/issues/406, where the code generator
silently dropped these constraints so generated models validated payloads that
factory-built models rejected.

Known blind spots:

- Fields identical to a base-class field (e.g. ``Element.id``, ``Resource.meta``)
  share literally the same ``FieldInfo`` object between the factory-built and the
  generated model, so the comparison trivially passes for inherited fields.  Only
  *redeclared* fields — those where the subclass overrides the base definition —
  are genuinely exercised.
- Scalar string ``ElementDefinition.maxLength`` constraints on repeating elements
  are not encoded at all: Pydantic cannot apply ``MaxLen`` to fhircraft primitive
  model types (they define no ``__len__``), so the per-element string-length
  constraint is a known, documented gap on both sides of the comparison.
"""

import pytest

from fhircraft.fhir.resources.datatypes.registry import get_registry
from fhircraft.fhir.resources.factory import FHIRModelFactory
from fhircraft.fhir.resources.generator import _extract_constraint_metadata

# Only resources and complex types are generator output; primitive types are
# hand-written and must never be compared against a factory build.
GENERATED_KINDS = ("resource", "complex-type")


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "checked-in datatypes tree not yet regenerated with constraint metadata; "
        "un-xfail after the surgical tree update"
    ),
    strict=True,
)
@pytest.mark.parametrize("release", ["R4", "R4B", "R5"])
def test_factory_vs_generated_constraint_equivalence(release: str) -> None:
    registry = get_registry(release)
    factory = FHIRModelFactory(release)
    mismatches: list[str] = []
    for kind in GENERATED_KINDS:
        for url in registry.all_urls(kind=kind):
            # Resolve the generated class BEFORE the factory build: get_by_url
            # gives priority to the factory construction cache, which would
            # otherwise return the factory-built model for both sides.
            generated_model = registry.get_by_url(url)
            if generated_model is None:
                continue
            factory_model = factory.build(canonical_url=url)
            for field_name, factory_info in factory_model.model_fields.items():
                generated_info = generated_model.model_fields.get(field_name)
                if generated_info is None:
                    # Field-set drift is a different defect; this test guards
                    # constraint metadata only.
                    continue
                factory_constraints = _extract_constraint_metadata(
                    factory_info.metadata
                )
                generated_constraints = _extract_constraint_metadata(
                    generated_info.metadata
                )
                if factory_constraints != generated_constraints:
                    mismatches.append(
                        f"{url} field={field_name!r}: "
                        f"factory={factory_constraints} "
                        f"generated={generated_constraints}"
                    )
    assert not mismatches, (
        f"{len(mismatches)} constraint-metadata mismatches between factory-built "
        f"and generated {release} models (first 25 shown):\n"
        + "\n".join(mismatches[:25])
    )
