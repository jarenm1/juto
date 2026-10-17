// Semantic selector engines (`label/`, `placeholder/`, `testid/`, `alt/`, `title/`, `role/`).
// Vendored from oh-my-pi packages/coding-agent/src/tools/browser/query-handlers.ts at
// 579da1d661c5cb8d43bc2ddd429ab72e67165ad8, MIT (see licenses/OMP-MIT.txt); TypeScript
// annotations removed. Evaluated as an object literal body bound to `semantic`.
label: function (node, selector) {
	const root = node;
	const wanted = selector.trim().toLocaleLowerCase();
	const matches = value => (value ?? "").trim().toLocaleLowerCase().includes(wanted);
	const elements = Array.from(root.querySelectorAll("*"));
	const result = [];
	const seen = new Set();
	const add = element => {
		if (element && !seen.has(element)) {
			seen.add(element);
			result.push(element);
		}
	};
	for (const label of root.querySelectorAll("label")) {
		if (!matches(label.textContent)) continue;
		const htmlFor = label.getAttribute("for");
		if (htmlFor) {
			const target = label.ownerDocument?.getElementById(htmlFor) ?? null;
			if (target && (node === target || node.contains(target))) {
				add(target);
			}
		} else {
			add(label.querySelector("button,input,meter,output,progress,select,textarea"));
		}
	}
	for (const element of elements) {
		if (matches(element.getAttribute("aria-label"))) add(element);
		const labelledBy = element.getAttribute("aria-labelledby");
		if (!labelledBy) continue;
		const labelText = labelledBy
			.split(/\s+/)
			.map(id => element.ownerDocument?.getElementById(id)?.textContent ?? "")
			.join(" ");
		if (matches(labelText)) add(element);
	}
	return result;
},
placeholder: function (node, selector) {
	const wanted = selector.trim().toLocaleLowerCase();
	return Array.from(node.querySelectorAll("[placeholder]")).filter(element =>
		(element.getAttribute("placeholder") ?? "").trim().toLocaleLowerCase().includes(wanted),
	);
},
testid: function (node, selector) {
	const wanted = selector.trim();
	return Array.from(node.querySelectorAll("[data-testid]")).filter(
		element => element.getAttribute("data-testid") === wanted,
	);
},
alt: function (node, selector) {
	const wanted = selector.trim().toLocaleLowerCase();
	return Array.from(node.querySelectorAll("[alt]")).filter(element =>
		(element.getAttribute("alt") ?? "").trim().toLocaleLowerCase().includes(wanted),
	);
},
title: function (node, selector) {
	const wanted = selector.trim().toLocaleLowerCase();
	return Array.from(node.querySelectorAll("[title]")).filter(element =>
		(element.getAttribute("title") ?? "").trim().toLocaleLowerCase().includes(wanted),
	);
},
role: function (node, selector) {
	const root = node;
	const roleMatch = /^\s*([^\s[]+)/.exec(selector);
	const wantedRole = (roleMatch?.[1] ?? "").toLocaleLowerCase();
	const nameMatch = /\[\s*name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\]\s]+))(?:\s+(exact))?\s*\]/i.exec(selector);
	const wantedName = (nameMatch?.[1] ?? nameMatch?.[2] ?? nameMatch?.[3])?.trim().toLocaleLowerCase();
	const exact = nameMatch?.[4]?.toLocaleLowerCase() === "exact";
	const implicitRole = element => {
		const tag = element.tagName.toLocaleLowerCase();
		if (tag === "a" && element.hasAttribute("href")) return "link";
		if (tag === "button") return "button";
		if (tag === "textarea") return "textbox";
		if (tag === "select") return element.multiple || element.size > 1 ? "listbox" : "combobox";
		if (tag === "option") return "option";
		if (tag === "img") return "img";
		if (tag === "ul" || tag === "ol") return "list";
		if (tag === "li") return "listitem";
		if (tag === "nav") return "navigation";
		if (tag === "main") return "main";
		if (tag === "form") return "form";
		if (tag === "article") return "article";
		if (/^h[1-6]$/.test(tag)) return "heading";
		if (tag === "table") return "table";
		if (tag === "tr") return "row";
		if (tag === "th") return element.getAttribute("scope") === "row" ? "rowheader" : "columnheader";
		if (tag === "td") return "cell";
		if (tag !== "input") return null;
		const type = (element.type || "text").toLocaleLowerCase();
		if (type === "checkbox") return "checkbox";
		if (type === "radio") return "radio";
		if (type === "range") return "slider";
		if (type === "number") return "spinbutton";
		if (type === "search") return "searchbox";
		if (["button", "submit", "reset", "image"].includes(type)) return "button";
		if (!["hidden", "file", "color", "date", "datetime-local", "month", "time", "week"].includes(type)) {
			return "textbox";
		}
		return null;
	};
	const accessibleName = element => {
		const ariaLabel = element.getAttribute("aria-label");
		if (ariaLabel) return ariaLabel.trim();
		const labelledBy = element.getAttribute("aria-labelledby");
		if (labelledBy) {
			const value = labelledBy
				.split(/\s+/)
				.map(id => element.ownerDocument?.getElementById(id)?.textContent ?? "")
				.join(" ")
				.trim();
			if (value) return value;
		}
		if (element.id) {
			for (const label of element.ownerDocument?.querySelectorAll("label") ?? []) {
				if (label.getAttribute("for") === element.id) return (label.textContent ?? "").trim();
			}
		}
		const wrappingLabel = element.closest("label");
		if (wrappingLabel) return (wrappingLabel.textContent ?? "").trim();
		return (
			element.getAttribute("alt") ??
			element.getAttribute("title") ??
			(element.tagName.toLocaleLowerCase() === "input" &&
			["button", "submit", "reset"].includes(element.type.toLocaleLowerCase())
				? element.value
				: element.textContent) ??
			element.getAttribute("placeholder") ??
			""
		).trim();
	};
	return Array.from(root.querySelectorAll("*")).filter(element => {
		const role = (element.getAttribute("role")?.trim().split(/\s+/)[0] ?? implicitRole(element))?.toLocaleLowerCase();
		if (role !== wantedRole) return false;
		if (wantedName === undefined) return true;
		const name = accessibleName(element).toLocaleLowerCase();
		return exact ? name === wantedName : name.includes(wantedName);
	});
},
