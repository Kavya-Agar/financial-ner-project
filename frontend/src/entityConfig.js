// Central registry of the 8 entity types the backend can return.
// Colors are chosen from a validated categorical palette (checked for
// colorblind-safe adjacent separation and >=4.5:1 text contrast against
// their own swatch background in both light and dark mode -- see
// App.css for the actual color values, which must stay in sync with
// this list of codes/labels).
export const ENTITY_ORDER = [
  'PER',
  'ORG',
  'LOC',
  'AMOUNT',
  'DATE',
  'ACCOUNT',
  'SSN',
  'FORM',
];

export const ENTITY_TYPES = {
  PER: { label: 'Person' },
  ORG: { label: 'Organization' },
  LOC: { label: 'Location' },
  AMOUNT: { label: 'Amount' },
  DATE: { label: 'Date' },
  ACCOUNT: { label: 'Account' },
  SSN: { label: 'SSN' },
  FORM: { label: 'Form' },
};

export function isKnownEntityLabel(label) {
  return Object.prototype.hasOwnProperty.call(ENTITY_TYPES, label);
}
