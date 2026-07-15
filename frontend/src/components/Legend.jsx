import { ENTITY_ORDER, ENTITY_TYPES } from '../entityConfig.js';

export default function Legend() {
  return (
    <div className="legend" aria-label="Entity type legend">
      {ENTITY_ORDER.map((code) => (
        <span key={code} className="legend-item" data-testid="legend-item">
          <span
            className={`legend-swatch entity-${code}`}
            aria-hidden="true"
          />
          {ENTITY_TYPES[code].label}
        </span>
      ))}
    </div>
  );
}
