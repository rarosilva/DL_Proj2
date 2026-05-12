import json
from pathlib import Path

p = Path("tasks/task1_organized.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
changed = False

for c in nb["cells"]:
    s = "".join(c.get("source", []))
    if "Environments created:" in s and "assert envs['difficulty_0'].action_space.n == 2" in s:
        bad = (
            "assert envs['difficulty_0'].action_space.n == 2, "
            "f\"unexpected action count: {envs['difficulty_0'].action_space.n}\"\n"
            "print(\"  [ok] env shapes confirmed\")"
        )
        good = (
            "    assert envs['difficulty_0'].action_space.n == 2, "
            "f\"unexpected action count: {envs['difficulty_0'].action_space.n}\"\n"
            "    print(\"  [ok] env shapes confirmed\")"
        )
        s = s.replace(bad, good)
        c["source"] = [ln + "\n" for ln in s.splitlines()]
        changed = True
        break

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("changed:", changed)
