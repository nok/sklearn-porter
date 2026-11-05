// Load a ported estimator or transformer of JavaScript as a module (not as a
// program) and print the results of its methods for the samples of the
// standard input (one per line) in the format of the program of the ported
// estimator, so that both outputs can be compared.
//
// Usage: node js_module.js <configuration>
//
// The configuration (JSON) contains:
// - `mode`: how the module is loaded: `browser` (in a context with only the
//   globals of a browser, without `require`, `module` and `process`),
//   `require` (a module of Node) or `import` (an ES module)
// - `src`: the path of the ported estimator
// - `className`: the name of the exported class
// - `data`: the path of the model data (template `exported`) or null
// - `methods`: the keys of the results (e.g. `predict` and `predict_proba`)
//   or `transform` (the values of a transformer); without methods only the
//   type of the export is printed
// - `documents`: whether the samples are documents (e.g. of a vectorizer)
'use strict';

const fs = require('fs');
const path = require('path');
const url = require('url');
const vm = require('vm');

const config = JSON.parse(process.argv[2]);
const src = path.resolve(config.src);

const load = async () => {
    if (config.mode === 'browser') {
        const context = vm.createContext({ console, TextEncoder });
        vm.runInContext(fs.readFileSync(src, 'utf8'), context, { filename: src });
        return context[config.className];
    }
    if (config.mode === 'require') {
        return require(src);
    }
    // An ES module (e.g. in a browser) needs the extension `.mjs` in Node:
    const copy = src.replace(/\.js$/, '.mjs');
    fs.copyFileSync(src, copy);
    await import(url.pathToFileURL(copy).href);
    return globalThis[config.className];
};

// Parse a feature like the program of the ported estimator:
const parseFeature = (value) => {
    const name = value.toLowerCase().replace(/^[+-]/, '');
    if (name === 'inf' || name === 'infinity') {
        return value.charAt(0) === '-' ? -Infinity : Infinity;
    }
    return name === 'nan' ? NaN : parseFloat(value);
};

const main = async () => {
    const Clazz = await load();
    if (config.methods.length === 0) {
        console.log(typeof Clazz);
        return;
    }
    let instance;
    if (config.data) {
        // Node reads the file itself, a browser gets the parsed model data:
        const data = config.mode === 'require'
            ? config.data
            : JSON.parse(fs.readFileSync(config.data, 'utf8'));
        instance = new Clazz(data);
    } else {
        instance = new Clazz();
    }
    const lines = fs.readFileSync(0, 'utf8').split('\n');
    if (lines[lines.length - 1] === '') {
        lines.pop();
    }
    const outputs = lines.map((line) => {
        const sample = config.documents
            ? line
            : line.trim().split(/\s+/).map(parseFeature);
        if (config.methods[0] === 'transform') {
            return instance.transform(sample).map((value) => {
                return Object.is(value, -0) ? '-0' : String(value);
            }).join(' ');
        }
        const results = {};
        config.methods.forEach((key) => {
            const method = key.replace(/_(\w)/g, (_, c) => c.toUpperCase());
            results[key] = instance[method](sample);
        });
        return JSON.stringify(results);
    });
    process.stdout.write(outputs.map((output) => output + '\n').join(''));
};

main();
