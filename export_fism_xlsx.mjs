// Optional Artifact Tool exporter; eval_fism.py falls back on servers without this runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
let library;
try {
  const entry = process.env.FISM_NODE_MODULES
    ? require.resolve('@oai/artifact-tool', { paths: [process.env.FISM_NODE_MODULES] })
    : require.resolve('@oai/artifact-tool');
  library = await import(pathToFileURL(entry).href);
} catch (error) {
  if (error.code === 'MODULE_NOT_FOUND' || error.code === 'ERR_MODULE_NOT_FOUND') process.exit(42);
  throw error;
}
const { Workbook, SpreadsheetFile } = library;
const [input, output] = process.argv.slice(2);
const { columns, rows } = JSON.parse(await fs.readFile(input, 'utf8'));
const book = Workbook.create();
const sheet = book.worksheets.add('All model tests');
sheet.showGridLines = false;
const safe = value => typeof value === 'string' && value.startsWith('=') ? "'" + value : value;
const data = [columns, ...rows.map(row => row.map(safe))];
const area = sheet.getRangeByIndexes(0, 0, data.length, columns.length);
area.values = data;
area.format.font = { name: 'Arial', size: 10, color: '#243B53' };
area.format.rowHeight = 22;
area.format.verticalAlignment = 'center';
const head = sheet.getRangeByIndexes(0, 0, 1, columns.length);
head.format.fill = '#243B53';
head.format.font = { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' };
head.format.wrapText = true;
head.format.horizontalAlignment = 'center';
head.format.rowHeight = 32;
columns.forEach((field, col) => {
  const range = sheet.getRangeByIndexes(0, col, data.length, 1);
  range.format.columnWidth = ['checkpoint', 'error'].includes(field) ? 55 : field === 'job_id' ? 42 : 22;
  if (rows.length && (field.includes('@') || ['fism_reg', 'lr'].includes(field))) {
    sheet.getRangeByIndexes(1, col, rows.length, 1).setNumberFormat('0.000000');
  }
});
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(3);
sheet.tables.add(area, true, 'FISMTests');
book.recalculate();
await fs.mkdir(path.dirname(path.resolve(output)), { recursive: true });
await (await SpreadsheetFile.exportXlsx(book)).save(output);
if (process.env.FISM_XLSX_PREVIEW) {
  const preview = await book.render({ sheetName: sheet.name, range: 'A1:L7', scale: 1.5, format: 'png' });
  await fs.writeFile(process.env.FISM_XLSX_PREVIEW, new Uint8Array(await preview.arrayBuffer()));
  console.log((await book.inspect({ kind: 'table', range: 'All model tests!A1:L4',
    include: 'values', tableMaxRows: 4, tableMaxCols: 12, maxChars: 2000 })).ndjson);
}
