import { FieldTooltip } from "./FieldTooltip";

interface RangeFieldProps {
  label: string;
  value: { Min: number; Max: number };
  description?: string;
  onChange: (value: { Min: number; Max: number }) => void;
  onBlur?: () => void;
}

export function RangeField({ label, value, description, onChange, onBlur }: RangeFieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-tn-text-muted flex items-center">
        {label}
        {description && <FieldTooltip description={description} />}
      </label>
      <div className="flex gap-1">
        {(["Min", "Max"] as const).map((bound) => (
          <div key={bound} className="flex-1">
            <label className="text-[10px] text-tn-text-muted uppercase">{bound}</label>
            <input
              type="number"
              value={value[bound]}
              onChange={(e) =>
                onChange({ ...value, [bound]: parseFloat(e.target.value) || 0 })
              }
              onBlur={onBlur}
              className="w-full px-1.5 py-0.5 text-xs bg-tn-bg border border-tn-border rounded"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
