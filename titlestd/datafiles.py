"""Readers and writers for the data files of docs/design.md 2.3.

Six of the seven files are handled here, with exactly the column names the
design lists. The seventh, the truth file, is read and written only in
titlestd.people, so that nothing else can reach it (2.4, Q22).

Format (Q35, Q38): UTF-8 CSV with a header row, fields quoted only where
needed, lines ending in a line feed. A flag is written as 1 or 0 and a missing
grade as an empty field. Nothing is trimmed or changed on reading: a job title
comes back byte for byte, which the reverse file relies on (Q05).

The writers exist for the data generator and for tests. A file written here,
read and written again, gives the same bytes.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from titlestd.contracts import (
    Abbreviation,
    Catalog,
    CatalogRole,
    Employee,
    GradeLevel,
    InputData,
    LadderLevel,
    SeedLookupRow,
    TitlestdError,
)

ROLES_FILE = "catalog_roles.csv"
ROLES_COLUMNS = (
    "role_id",
    "family_code",
    "role_name",
    "short_name",
    "definition",
    "compliance_flag",
    "is_generic",
)
LADDER_FILE = "level_ladder.csv"
LADDER_COLUMNS = ("level", "long_word", "short_word")
GRADES_FILE = "grade_levels.csv"
GRADES_COLUMNS = ("grade", "level")
ABBREVIATIONS_FILE = "abbreviations.csv"
ABBREVIATIONS_COLUMNS = ("short_form", "long_form")
EMPLOYEES_FILE = "employees.csv"
EMPLOYEES_COLUMNS = (
    "employee_id",
    "job_title",
    "job_family_code",
    "department",
    "grade",
    "job_description",
)
SEED_LOOKUP_FILE = "seed_lookup.csv"
SEED_LOOKUP_COLUMNS = ("job_title", "job_family_code", "role_id")

# The files the program itself reads, in a fixed order.
INPUT_FILES = (
    ROLES_FILE,
    LADDER_FILE,
    GRADES_FILE,
    ABBREVIATIONS_FILE,
    EMPLOYEES_FILE,
    SEED_LOOKUP_FILE,
)


class DataError(TitlestdError):
    """A data file is missing, malformed or inconsistent with another."""


# --- Plain tables -------------------------------------------------------------


def read_table(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read a CSV file whose header is exactly columns, as one dict per row."""
    return [row for _, row in _numbered_rows(Path(path), columns)]


def write_table(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    """Write rows under the header columns. Each row must have exactly those keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            if set(row) != set(columns):
                raise DataError(
                    f"{path.name}: a row has columns {sorted(row)}, expected {list(columns)}"
                )
            writer.writerow([row[column] for column in columns])


def _numbered_rows(path: Path, columns: Sequence[str]) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield (line number, row) for every data row, checking header and width."""
    try:
        handle = path.open(encoding="utf-8", newline="")
    except FileNotFoundError:
        raise DataError(f"data file not found: {path}") from None
    except OSError as error:
        raise DataError(f"data file cannot be read: {path}: {error}") from None
    with handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader, None)
            if header != list(columns):
                raise DataError(
                    f"{path.name}: the header must be {','.join(columns)}; "
                    f"found {','.join(header) if header else 'an empty file'}"
                )
            for fields in reader:
                if not fields:
                    continue
                if len(fields) != len(columns):
                    raise DataError(
                        f"{path.name} line {reader.line_num}: "
                        f"{len(fields)} fields, expected {len(columns)}"
                    )
                yield reader.line_num, dict(zip(columns, fields, strict=True))
        except (csv.Error, UnicodeDecodeError) as error:
            raise DataError(f"{path.name}: cannot be read as UTF-8 CSV: {error}") from None


def _required(row: Mapping[str, str], column: str, name: str, line: int) -> str:
    value = row[column]
    if not value.strip():
        raise DataError(f"{name} line {line}: {column} is empty")
    return value


def _flag(row: Mapping[str, str], column: str, name: str, line: int) -> bool:
    value = row[column]
    if value not in ("0", "1"):
        raise DataError(f"{name} line {line}: {column} must be 0 or 1, found {value!r}")
    return value == "1"


def _whole_number(row: Mapping[str, str], column: str, name: str, line: int) -> int:
    value = row[column]
    if not (value.isascii() and value.isdigit()):
        raise DataError(f"{name} line {line}: {column} must be a whole number, found {value!r}")
    return int(value)


def _unique(seen: set, key: object, what: str, name: str, line: int) -> None:
    if key in seen:
        raise DataError(f"{name} line {line}: {what} appears more than once")
    seen.add(key)


# --- One reader and one writer per file ---------------------------------------


def read_catalog_roles(data_dir: Path) -> tuple[CatalogRole, ...]:
    roles, seen = [], set()
    for line, row in _numbered_rows(Path(data_dir) / ROLES_FILE, ROLES_COLUMNS):
        role = CatalogRole(
            role_id=_required(row, "role_id", ROLES_FILE, line),
            family_code=_required(row, "family_code", ROLES_FILE, line),
            role_name=_required(row, "role_name", ROLES_FILE, line),
            short_name=_required(row, "short_name", ROLES_FILE, line),
            definition=row["definition"],
            compliance_flag=_flag(row, "compliance_flag", ROLES_FILE, line),
            is_generic=_flag(row, "is_generic", ROLES_FILE, line),
        )
        _unique(seen, role.role_id, f"role_id {role.role_id}", ROLES_FILE, line)
        roles.append(role)
    return tuple(roles)


def write_catalog_roles(data_dir: Path, roles: Iterable[CatalogRole]) -> None:
    write_table(
        Path(data_dir) / ROLES_FILE,
        ROLES_COLUMNS,
        (
            {
                "role_id": role.role_id,
                "family_code": role.family_code,
                "role_name": role.role_name,
                "short_name": role.short_name,
                "definition": role.definition,
                "compliance_flag": "1" if role.compliance_flag else "0",
                "is_generic": "1" if role.is_generic else "0",
            }
            for role in roles
        ),
    )


def read_level_ladder(data_dir: Path) -> tuple[LadderLevel, ...]:
    ladder, seen = [], set()
    for line, row in _numbered_rows(Path(data_dir) / LADDER_FILE, LADDER_COLUMNS):
        level = LadderLevel(
            level=_required(row, "level", LADDER_FILE, line),
            long_word=_required(row, "long_word", LADDER_FILE, line),
            short_word=_required(row, "short_word", LADDER_FILE, line),
        )
        _unique(seen, level.level, f"level {level.level}", LADDER_FILE, line)
        ladder.append(level)
    return tuple(ladder)


def write_level_ladder(data_dir: Path, ladder: Iterable[LadderLevel]) -> None:
    write_table(
        Path(data_dir) / LADDER_FILE,
        LADDER_COLUMNS,
        (
            {"level": level.level, "long_word": level.long_word, "short_word": level.short_word}
            for level in ladder
        ),
    )


def read_grade_levels(data_dir: Path) -> tuple[GradeLevel, ...]:
    grades, seen = [], set()
    for line, row in _numbered_rows(Path(data_dir) / GRADES_FILE, GRADES_COLUMNS):
        grade = GradeLevel(
            grade=_whole_number(row, "grade", GRADES_FILE, line),
            level=_required(row, "level", GRADES_FILE, line),
        )
        _unique(seen, grade.grade, f"grade {grade.grade}", GRADES_FILE, line)
        grades.append(grade)
    return tuple(grades)


def write_grade_levels(data_dir: Path, grade_levels: Iterable[GradeLevel]) -> None:
    write_table(
        Path(data_dir) / GRADES_FILE,
        GRADES_COLUMNS,
        ({"grade": str(grade.grade), "level": grade.level} for grade in grade_levels),
    )


def read_abbreviations(data_dir: Path) -> tuple[Abbreviation, ...]:
    abbreviations, seen = [], set()
    name = ABBREVIATIONS_FILE
    for line, row in _numbered_rows(Path(data_dir) / name, ABBREVIATIONS_COLUMNS):
        abbreviation = Abbreviation(
            short_form=_required(row, "short_form", name, line),
            long_form=_required(row, "long_form", name, line),
        )
        _unique(seen, abbreviation.short_form, f"short_form {abbreviation.short_form}", name, line)
        abbreviations.append(abbreviation)
    return tuple(abbreviations)


def write_abbreviations(data_dir: Path, abbreviations: Iterable[Abbreviation]) -> None:
    write_table(
        Path(data_dir) / ABBREVIATIONS_FILE,
        ABBREVIATIONS_COLUMNS,
        ({"short_form": a.short_form, "long_form": a.long_form} for a in abbreviations),
    )


def read_employees(data_dir: Path) -> tuple[Employee, ...]:
    employees, seen = [], set()
    for line, row in _numbered_rows(Path(data_dir) / EMPLOYEES_FILE, EMPLOYEES_COLUMNS):
        employee = Employee(
            employee_id=_required(row, "employee_id", EMPLOYEES_FILE, line),
            job_title=_required(row, "job_title", EMPLOYEES_FILE, line),
            job_family_code=row["job_family_code"],
            department=row["department"],
            grade=_whole_number(row, "grade", EMPLOYEES_FILE, line) if row["grade"] else None,
            job_description=row["job_description"],
        )
        what = f"employee_id {employee.employee_id}"
        _unique(seen, employee.employee_id, what, EMPLOYEES_FILE, line)
        employees.append(employee)
    return tuple(employees)


def write_employees(data_dir: Path, employees: Iterable[Employee]) -> None:
    write_table(
        Path(data_dir) / EMPLOYEES_FILE,
        EMPLOYEES_COLUMNS,
        (
            {
                "employee_id": employee.employee_id,
                "job_title": employee.job_title,
                "job_family_code": employee.job_family_code,
                "department": employee.department,
                "grade": "" if employee.grade is None else str(employee.grade),
                "job_description": employee.job_description,
            }
            for employee in employees
        ),
    )


def read_seed_lookup(data_dir: Path) -> tuple[SeedLookupRow, ...]:
    rows, seen = [], set()
    name = SEED_LOOKUP_FILE
    for line, row in _numbered_rows(Path(data_dir) / name, SEED_LOOKUP_COLUMNS):
        seed = SeedLookupRow(
            job_title=_required(row, "job_title", name, line),
            job_family_code=row["job_family_code"],
            role_id=_required(row, "role_id", name, line),
        )
        pair = (seed.job_title, seed.job_family_code)
        _unique(
            seen, pair, f"title {seed.job_title!r} in family {seed.job_family_code!r}", name, line
        )
        rows.append(seed)
    return tuple(rows)


def write_seed_lookup(data_dir: Path, rows: Iterable[SeedLookupRow]) -> None:
    write_table(
        Path(data_dir) / SEED_LOOKUP_FILE,
        SEED_LOOKUP_COLUMNS,
        (
            {
                "job_title": row.job_title,
                "job_family_code": row.job_family_code,
                "role_id": row.role_id,
            }
            for row in rows
        ),
    )


# --- The six input files together ---------------------------------------------


def read_catalog(data_dir: Path) -> Catalog:
    return Catalog(
        roles=read_catalog_roles(data_dir),
        ladder=read_level_ladder(data_dir),
        grade_levels=read_grade_levels(data_dir),
    )


def read_inputs(data_dir: Path) -> InputData:
    """Read the six input files and check them against each other."""
    data = InputData(
        catalog=read_catalog(data_dir),
        abbreviations=read_abbreviations(data_dir),
        employees=read_employees(data_dir),
        seed_lookup=read_seed_lookup(data_dir),
    )
    validate_inputs(data)
    return data


def write_inputs(data_dir: Path, data: InputData) -> None:
    write_catalog_roles(data_dir, data.catalog.roles)
    write_level_ladder(data_dir, data.catalog.ladder)
    write_grade_levels(data_dir, data.catalog.grade_levels)
    write_abbreviations(data_dir, data.abbreviations)
    write_employees(data_dir, data.employees)
    write_seed_lookup(data_dir, data.seed_lookup)


def validate_inputs(data: InputData) -> None:
    """Check what one file cannot check alone. Raises DataError on the first fault."""
    catalog = data.catalog
    if not catalog.roles:
        raise DataError(f"{ROLES_FILE}: the catalog has no roles")
    families = sorted({role.family_code for role in catalog.roles})
    for family in families:
        generic = [r for r in catalog.roles if r.family_code == family and r.is_generic]
        if len(generic) != 1:
            raise DataError(
                f"{ROLES_FILE}: family {family} must have exactly one generic role, "
                f"found {len(generic)}"
            )
    levels = {level.level for level in catalog.ladder}
    for grade in catalog.grade_levels:
        if grade.level not in levels:
            raise DataError(
                f"{GRADES_FILE}: grade {grade.grade} names level {grade.level}, "
                f"which is not in {LADDER_FILE}"
            )
    check_decided_roles(
        SEED_LOOKUP_FILE,
        catalog,
        ((row.job_title, row.job_family_code, row.role_id) for row in data.seed_lookup),
    )


def check_decided_roles(name: str, catalog: Catalog, rows: Iterable[tuple[str, str, str]]) -> None:
    """Check rows of (job title, family code as recorded, role id) a person decided.

    The role must exist and must not be generic; where the family code is one
    of the catalog's, the role must be in that family (Q09). Used for the seed
    look-up here and for the truth file in titlestd.people.
    """
    by_id = {role.role_id: role for role in catalog.roles}
    families = {role.family_code for role in catalog.roles}
    for job_title, family_code, role_id in rows:
        role = by_id.get(role_id)
        where = f"{name}: title {job_title!r} in family {family_code!r}"
        if role is None:
            raise DataError(f"{where} names role {role_id}, which is not in the catalog")
        if role.is_generic:
            raise DataError(f"{where} names the generic role {role_id}; only code assigns it")
        family = family_code.strip()
        if family in families and role.family_code != family:
            raise DataError(f"{where} names role {role_id} of family {role.family_code}")


# --- Fingerprints (Q18, Q29) --------------------------------------------------


def fingerprint_files(paths: Iterable[Path]) -> str:
    """SHA-256 over the names and bytes of the files, in the order given."""
    digest = hashlib.sha256()
    for path in paths:
        try:
            content = Path(path).read_bytes()
        except FileNotFoundError:
            raise DataError(f"data file not found: {path}") from None
        except OSError as error:
            raise DataError(f"data file cannot be read: {path}: {error}") from None
        digest.update(f"{Path(path).name}\n{len(content)}\n".encode())
        digest.update(content)
    return digest.hexdigest()


def input_fingerprint(data_dir: Path) -> str:
    """Fingerprint of the six input files, for a recording's manifest."""
    return fingerprint_files(Path(data_dir) / name for name in INPUT_FILES)
