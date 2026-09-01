# Dataset records

A dataset record table is a parquet file with one row per annotated sample. It is generated
outside this repository and read here. The generator owns the annotations, the taxonomy, the
sweeps and the splits. This repository owns loading and training, and nothing crosses that
line.

## Ownership

| Concern | Owner |
| --- | --- |
| Reading the dataset annotations | the generator |
| Resolving raw labels into trained classes | the generator |
| Choosing sweeps per sample | the generator |
| Assigning scenarios to train, val and test | the generator |
| Selecting databases and splits for a run | this repository |
| Loading points, boxes and masks into samples | this repository |

For T4dataset the generator is
[t4dataset-generator](https://github.com/tier4/t4dataset-generator). nuScenes has no external
generator, so its table is written here, see [nuScenes](#nuscenes).

## Reading a table

`RecordTable` is the whole read side:

```python
from autoware_ml.databases.record_table import RecordTable

table = RecordTable(
    path="/workspace/records/t4dataset.parquet",
    data_root="/workspace/data/t4dataset/",
    databases=["db_j6gen2_v1"],
)
records = table.load("train")
```

- `path` is the parquet file. Nothing about a training configuration enters its name, so one
  table is shared by every model that trains on its databases.
- `data_root` is the read only data mount the record paths resolve against.
- `databases` narrows the table, or is empty to keep everything.
- `load(split)` returns the rows of one split, ordered by scenario and sample, and raises when
  the split holds no records or a named database is absent.

Tables live at a fixed `/workspace/records`, a separate writable mount, so the data mount can
stay read only:

```bash
./docker/container.sh --records-path /my/records   # or AUTOWARE_ML_RECORDS_PATH
```

The `configs/records/` group holds one config per corpus, see
[configuration](../user-guide/configuration.md).

## Splits and selection

Every row carries `database` and `split`, so a datamodule selects data with a filter and holds
no scenario list of its own. Splits are decided by the generator, which means the frames of one
scenario cannot straddle a split boundary.

A run may mix several sources over the same or different tables. Each source declares its own
supervision coverage and repeat factor, so a pseudo labelled corpus and a rehearsal corpus can
train together.

## Schema

`autoware_ml/databases/schemas/` defines the table, see [schemas](schemas.md). A record loads
back into `DatasetRecord` with `DatasetRecord.load_from_dictionary(row)`, which is the contract a
generator has to satisfy.

## nuScenes

nuScenes records are written here because no external generator exists:

```bash
python3 autoware_ml/scripts/generate_nuscenes_records.py --config-name nuscenes_records
```

`NuscenesRecordsWriter` builds the records with the devkit and writes them to the `out_file` of
`configs/writers/nuscenes_records.yaml`. Its scenarios come from the official devkit scene
splits and stamp their split onto every record.

## Files

| Path | Purpose |
| --- | --- |
| `autoware_ml/databases/record_table.py` | read access to a table |
| `autoware_ml/databases/schemas/` | table and nested data model definitions |
| `autoware_ml/databases/nuscenes/` | the nuScenes records writer and its generator |
| `autoware_ml/scripts/generate_nuscenes_records.py` | entrypoint for the nuScenes table |
| `autoware_ml/configs/records/` | one record table config per corpus |
| `autoware_ml/configs/writers/` | the nuScenes writer config |
