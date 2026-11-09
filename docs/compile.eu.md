# Urratsez urrats konpilatzea

[In English: compile.md](compile.md)

`qoa compile`-k programa bat bere `.qx` bytecode bihurtzen du. `--emit`-ekin makina-koderako bideko edozein
urratsetan geldi daiteke eta urrats hori idatzi, urrats bakoitzak zure programarekin zer egiten duen ikusteko,
urratsak alderatzeko edo bat arazteko:

```sh
qoa compile programa.qoa --emit qoa              # 1. QOA testu kanonikoa
qoa compile programa.qoa --emit qx               # 2. bytecode-a (lehenetsia)
qoa compile programa.qoa --emit ir               # 3. oinarrizko blokeak, eragiketa bakoitza natiboa ala QOA-ren makinaren bidez
qoa compile programa.qoa --emit asm --arch s390x # 4. arkitektura baten mihiztadura
qoa compile programa.qoa --emit obj              # 5. makina-kodea: objektu-fitxategi bat (.o)
qoa compile programa.qoa --emit so               # 6. liburutegi partekatu bat (.so) ...
qoa run programa.so                              #    ... qoa-k exekutatzen duena, emaitza berberekin
```

Testuzko urratsak (`qoa`, `ir`, `asm`) irteera estandarrera doaz `-o FITXATEGIA` ematen ez baduzu; `obj` eta
`so` iturburuaren ondoan idazten dira (`programa.o`, `programa.so`) edo `-o`-n. `.qx` bat ere sarrera izan
daiteke.

## Arkitekturak

| `--arch` | prozesadoreak | mihiztadura-sintaxia |
|---|---|---|
| `x86_64` | Intel eta AMD 64 bit | AT&T |
| `aarch64` | ARM 64 bit (Graviton, Ampere, Apple txipak linux-en, Raspberry Pi 4/5) | ARM |
| `riscv64` | RISC-V 64 bit (RV64GC) | RISC-V |
| `powerpc64le` | IBM POWER8 eta berriagoak, little-endian | POWER (ELFv2) |
| `s390x` | IBM Z (z10 eta berriagoak) | z/Architecture |
| `loongarch64` | Loongson LoongArch | LoongArch |

`--arch` gabe, urratsa zauden makinarako egiten da. Kodea ELF sistemetarako da (linux eta antzekoak); macos-ek
eta windows-ek beste objektu-formatu batzuk erabiltzen dituzte, oraindik idazten ez direnak, beraz `--emit
obj`-ek eta `so`-k hala esaten dute han, eta `--emit asm --arch ...` edonon dabil.

`obj`-ek eta `so`-k C konpilatzaile bat behar dute mihiztatzeko eta estekatzeko: `cc` makina honetarako, eta
beste arkitektura baterako `clang --target=...` (`lld`-rekin `so`-rentzat) edo arkitektura horren GCC-a
(`aarch64-linux-gnu-gcc` eta abar). `--cc KOMANDOA`-k bat izendatzen du, adibidez `--cc "clang
--target=riscv64-linux-gnu -fuse-ld=lld"`. Bat ere aurkitzen ez bada, QOA-k esaten du zer instalatu.

## Kode natiboak zer egiten duen

IR-ak plana erakusten du. Agindu klasiko gutxi batzuk makina-agindu bihurtzen dira: `SET`, `COPY`, `ADD`,
`SUB` eta `MUL` erregistroekin edo zenbakiekin, eta `JUMP`. Gainerako eragiketa guztiak (ateak, optika,
neurketak, gainerako agindu klasikoak, sarrera eta irteera) QOA-ren makinari egindako deiak dira: hark
interpretatzaileak bezala exekutatzen du eta hurrengo eragiketa itzultzen du. Kode natiboa zuzenean jarraitzen
du hurrengoa ondokoa bada, eta bestela taula batean begiratzen du nora joan, beraz jauziek, deiek, itzulerek
eta begiztek funtzionatzen dute.

```
; 8 eragiketa 5 blokeetan: 2 natibo (makina-aginduak), 6 QOA-ren makinaren bidez (step)
b0:  ; eragiketak 0..1 -> b4
  0000  native   R0 = 5.0                  ; REGSET R0 5  (3. lerroa)
  0001  step     1                         ; CALL addr_0x6  (4. lerroa)
```

Batuketa, kenketa eta biderketa IEEE 754 eragiketa hutsak dira, beraz bit berberak ematen dituzte sei
arkitekturetan interpretatzailean bezala. Agindu natibo bakoitzaren aurretik, kodeak makinari esaten dio
(`tick`), beraz urrats-kontaketa, `--max-steps` eta programako posizioaren menpe dagoen guztia berdina da.
Ondorioz exekuzio natibo batek interpretatzailearen exekuzioa ematen du zehazki: kontaketa berberak hazi
berarekin, erregistro, irteera, errore eta lerro-zenbaki berberak. Probek hau egiaztatzen dute adibide eta
proba-programa guztietan, eta QOAn idatzitako QOA-ren mihiztatzailearekin, kode natibo gisa bere bytecode-a
bytez byte berreraikitzen duena; ci-ak sei arkitekturetan exekutatzen ditu (horietako bost qemu-n).

Liburutegiak bere programa darama (`.qx`-a, kargatzean bere SHA-256-arekin egiaztatuta), beraz `qoa run
programa.so`-k ez du beste ezer behar. `qoa run`-en aukera guztiek balio dute harekin: shot-ak, haziak, zarata,
`--mpi`, GPUak.

## Dei-hitzarmena

Mihiztadura irakurtzen edo beste nonbaitetik deitzen duenarentzat:

```c
// 0 itzultzen du programa amaitzean, 1 errore batean (makinak gordeta), 2 kodea eta programa bat ez badatoz
int64_t qoa_native_entry(void *machine, double regs[32],
                         int64_t (*step)(void *machine, uint64_t pc),   // hurrengo pc, -1 gelditu, -2 errorea
                         int64_t (*tick)(void *machine, uint64_t pc));  // 0, edo -2 errorea
extern const uint8_t  qoa_native_qx[];      // programaren .qx-a
extern const uint64_t qoa_native_qx_len;
extern const uint32_t qoa_native_abi;       // 1
```

`N` eragiketa bakoitzak `.Lop_N` etiketa du, eta `.Ltab` haien desplazamenduen taula da. Kodeak ez du kanpoko
ikurrik erabiltzen, beraz liburutegia posizioarekiko independentea da eta edonon kargatzen da.

## Segurtasuna

`.so` bat makina-kodea da: kargatzeak exekutatzen du. Zuk eraikitako liburutegiak bakarrik exekutatu, beste
edozein programa bezala. QOA-k egiaztatzen du liburutegi bat `qoa compile`-k egin zuela hitzarmenaren bertsio
honetarako eta haren programa osorik dagoela, baina ezin du eskuz editatutako makina-kodea egiaztatu.
Liburutegiari deitzen dion rust kodea `native` modulua da, `unsafe` erabil dezaketen gutxietako bat
([safety.eu.md](safety.eu.md)).
