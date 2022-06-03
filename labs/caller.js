const dylib = Deno.dlopen('./liba.a', {
    "numLogicalCpus": { parameters: ["i32"], result: "i32" },
  });
console.log(dylib.symbols.numLogicalCpus(2))