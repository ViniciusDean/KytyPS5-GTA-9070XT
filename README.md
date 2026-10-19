# KytyPS5-GTA-9070XT

Fork de testes do [TheCruZ/KytyPS5-GTA](https://github.com/TheCruZ/KytyPS5-GTA) para rodar
**GTA V de PS5 (PPSA04263)** no **KytyPS5** com **AMD RX 9070 XT (RDNA4)** e CPU Intel híbrida.
O foco é o modo **Performance RT** com ray tracing por hardware.

> **English summary:** experimental test fork of TheCruZ/KytyPS5-GTA tuned and measured on an
> AMD RX 9070 XT + Intel i5-14600K. It stacks TheCruZ's native hardware ray tracing branch, his v12
> shader work and performance commits from other forks, and adds an asynchronous guest fault
> readback (based on carbonimax's idea, rewritten for this fork's multi-threaded GPU pipeline).
> Driving in Performance RT went from ~16 to ~26 fps (median) so far. Builds are not published;
> the branches are for testing only.

> [!IMPORTANT]
> Builds experimentais, feitos e testados numa única máquina. Podem travar ou mostrar defeitos
> gráficos. Para jogar de forma estável, use os
> [**releases do TheCruZ**](https://github.com/TheCruZ/KytyPS5-GTA/releases).
>
> KytyPS5 não tem relação com a Sony Interactive Entertainment, PlayStation ou Rockstar Games.
> Este repositório não distribui jogos nem software do sistema; use só arquivos obtidos
> legalmente.

O README original, com o estado do jogo e as correções do TheCruZ, está
[aqui](https://github.com/TheCruZ/KytyPS5-GTA#readme). Para o emulador em geral, veja o
[README do KytyPS5](https://github.com/KytyPS5/KytyPS5#readme).

## Máquina de teste

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT (Adrenalin 32.0.32015.2008) |
| CPU | Intel Core i5-14600K (6 P-cores + 8 E-cores) |
| RAM | 32 GB |
| Sistema | Windows 11 Pro |
| Resolução | 2560x1440, tela cheia |

## Branches

O **`main` deste fork tem tudo junto**: é o `hwrt-v12-readback4` (RT por hardware do TheCruZ +
v12 + todos os commits de desempenho abaixo + readback assíncrono + compilação fora da thread de execução) com este README. Ele **não**
acompanha mais o `main` do TheCruZ. Os branches abaixo guardam cada etapa, do mais antigo para o
mais novo:

| Branch | Base | O que tem |
|---|---|---|
| [`hwrt`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt) | `native-hardware-raytracing` do TheCruZ (95ba8da) | RT por hardware (`VK_KHR_ray_query`) antes da v12 |
| [`gta-perf`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/gta-perf) | `hwrt` | correções de desempenho de outros forks (lista abaixo) |
| [`gta-perf2`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/gta-perf2) | `gta-perf` | + cache de pipelines do carbonimax, #1285 do NeySlim, bitmap BDA com um único atomic OR |
| [`hwrt-v12`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt-v12) | RT por hardware rebaseado (829bbda) | + os 7 commits de shader da v12 (EmK530) |
| [`hwrt-v12-perf`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt-v12-perf) | `hwrt-v12` | + o que funcionou no `gta-perf`/`gta-perf2` e a saída do EcoQoS |
| [`hwrt-v12-readback`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt-v12-readback) | `hwrt-v12-perf` | + **readback assíncrono** das faltas do jogo (melhor resultado até agora) |
| [`hwrt-v12-readback2`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt-v12-readback2) | `hwrt-v12-readback` | + estatística por motivo, até 32 cópias pendentes, flush do command buffer e faltas de escrita assíncronas |
| [`hwrt-v12-readback3`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt-v12-readback3) | `hwrt-v12-readback2` | + estatísticas de shader (DeepSeek), compilação de shader fora da thread de execução e draws adiados enquanto compila, origem dos `gpu-read` |
| [`hwrt-v12-readback4`](https://github.com/ViniciusDean/KytyPS5-GTA-9070XT/tree/hwrt-v12-readback4) | `hwrt-v12-readback3` | + log de shaders lentos (`KYTY_SLOW_SHADER_MS`) e a thread de execução não espera workers que compilam outra permutação do mesmo programa (em teste) — **é o `main`** |

Commits de outros autores mantêm o autor original. O que foi trazido para cá:

- **AN0DIN**: o event flag acorda só as threads que ele libera (o GTA acordava threads à toa
  ~50 mil vezes por segundo).
- **romainhedouin**: não faz a operação travada de leitura e escrita quando não há sinal pendente.
- **NeySlim**: partes do #1284 e do #1286; o #1285 (checagem de leituras escalares com intervalos
  de escrita mesclados e busca binária).
- **carbonimax**: o cache de pipelines não é regravado sem mudança nem perdido se a troca do
  arquivo falhar; a ideia do readback assíncrono.
- **nmzik**: cada palavra de PM4 `WRITE_DATA` é publicada uma única vez.
- **fxpw / TheCruZ**: bit do bitmap de faltas BDA marcado com um único atomic OR.
- **EmK530** (via v12 do TheCruZ): F16 nativo, leitura de constantes por uniform buffer, menos
  instruções desperdiçadas.

## Medições

Performance RT, dirigindo pela cidade por uns 2 minutos. O fps vem do título da janela do
emulador, amostrado a cada ~2 s; só entram amostras com a GPU acima de 30% (fora menus e pausa).
As cenas não são idênticas entre as partidas, então diferenças de 2 a 3 fps estão dentro do ruído.

| Build | fps médio | Mediana | 10% piores | GPU | CPU total |
|---|---|---|---|---|---|
| `hwrt-v12` | 15,7 | 16 | 12 | 58% | 23% |
| `hwrt-v12-perf` | 18–21 | 17–20 | 12–13 | 61–67% | 24–25% |
| `hwrt-v12-perf` + `KYTY_GPU_SPIN_US=8000` | 21–24 | 22–23 | 15–18 | 46–54% | 32–35% |
| `hwrt-v12-readback` (com spin 8000) | **27,4** | **26** | **17** | 55% | 35% |

Para comparação, o `hwrt` antigo dava ~19 fps em Performance RT e ~60 fps em Performance (sem RT).
Com o carro parado, o Performance RT fica entre 35 e 60 fps, com a GPU entre 70 e 88%.

## O que descobrimos

- **Dirigindo, o limite não é a GPU.** Ela fica em 45–60% e nenhuma thread do emulador chega a
  100% de um núcleo: a mais ocupada fica em ~80%. O tempo vai em **esperas de sincronização**
  entre as threads do emulador e a GPU, enquanto o jogo carrega a cidade.
- **A compilação de shaders não é a causa.** Com o cache aquecido, entram só 2 a 8 shaders novos
  por minuto, sem relação com as quedas.
- **Readback síncrono é uma das esperas.** Quando o jogo lê memória que a GPU escreveu, a thread de
  execução da GPU parava até a cópia terminar. O `hwrt-v12-readback` faz a cópia sem bloquear essa
  thread, e quem espera é só a thread do jogo que pediu os dados. Numa sessão de 5 minutos:
  - ~14 mil leituras foram pelo caminho novo (só 220 descartadas e refeitas);
  - ~57 mil (80%) ainda foram síncronas e custaram 16,4 s à thread de execução, chegando a
    ~21% do tempo dela nos piores trechos.
- **Spin mais longo ajuda um pouco.** Com `KYTY_GPU_SPIN_US=8000` as threads da GPU giram mais
  antes de dormir no kernel, o que reduz a latência entre elas, ao custo de mais CPU.

## Foco atual

1. **Levar mais leituras para o caminho assíncrono** (`hwrt-v12-readback2`): contar o motivo de
   cada readback síncrono (falta de escrita, falta do processador de comandos, limite de 8
   pendentes, escritas ainda sendo gravadas…) e ampliar o caminho novo para os mais comuns,
   sem nunca publicar dado velho.
2. **Travadas nas transições e no pause (~18 s):** nas trocas de área o emulador compilava ~136
   shaders seguidos na thread de execução (~96 ms cada). O `hwrt-v12-readback3` tira essa
   compilação da thread; no pause, a thread de execução chegou a esperar ~21 s por 6 shaders que
   compilavam nos workers; o `hwrt-v12-readback4` registra quais são e para de esperar os de
   outra permutação.
3. **Medir o que ainda não foi medido:**
   - a saída do EcoQoS, que nos testes acima estava desligada (`KYTY_POWER_THROTTLING=1`);
   - o log em arquivo (`--printf-direction File` grava ~1 MB/s nas próprias threads que enviam
     comandos para a GPU).
4. **Acompanhar o TheCruZ:** rebasear quando o RT por hardware ou a v12 avançarem, evitando os
   commits do upstream suspeitos da regressão das issues
   [#1591](https://github.com/KytyPS5/KytyPS5/issues/1591) e
   [#1626](https://github.com/KytyPS5/KytyPS5/issues/1626).

## Variáveis de ambiente

| Variável | Padrão | Efeito |
|---|---|---|
| `KYTY_SLOW_SHADER_MS` | 500 | registra no console compilações e esperas de shader acima desse tempo (0 desliga) |
| `KYTY_SHADER_WAIT_OTHERS=1` | desligado | volta a esperar workers que compilam outra permutação do mesmo programa |
| `KYTY_ASYNC_SHADERS=0` | ligado | não adia draws enquanto um shader compila (volta a compilar na thread de execução) |
| `KYTY_SHADER_STEAL=0`, `KYTY_SHADER_JOB_PRIORITY=0` | ligados | thread de execução não compila o próprio job / shaders da sessão não passam na frente do store |
| `KYTY_SHADER_STATS=1` | desligado | imprime a cada 5 s as estatísticas de shader no console |
| `KYTY_ASYNC_READBACK=0` | ligado | desliga o readback assíncrono (volta ao comportamento antigo) |
| `KYTY_READBACK_STATS=1` | desligado | imprime a cada 5 s uma linha `Readback stats:` no console |
| `KYTY_POWER_THROTTLING=1` | desligado | mantém o power throttling (EcoQoS) do Windows; por padrão o emulador sai dele |
| `KYTY_GPU_SPIN_US` | 2000 | quanto tempo (µs) as threads da GPU giram antes de dormir; testado com 8000 |
| `KYTY_GPU_THREAD_CPUS` | 4,6,8,10,2 | núcleos das threads da GPU; `off` deixa o Windows escolher |
| `KYTY_WAVE64_WARPS`, `KYTY_NATIVE_F16`, `KYTY_UNIFORM_BUFFERS` | ligados | chaves da v12 (EmK530); `=0` desliga cada uma |

## Avisos

- Não troque de modo gráfico com o jogo rodando: há relato de `ErrorDeviceLost` ao passar de
  Performance RT para Performance.
- O RT por hardware do TheCruZ ainda é experimental e pode ter falhas nas sombras.
- Cada build mantém a própria lista de shaders em `_PipelineCache`, então a primeira partida
  de um build novo engasga mais.

## Créditos

[TheCruZ](https://github.com/TheCruZ/KytyPS5-GTA), [KytyPS5](https://github.com/KytyPS5/KytyPS5)
e os autores dos commits trazidos de outros forks: AN0DIN, carbonimax, EmK530, fxpw, NeySlim,
nmzik e romainhedouin. As estatísticas de shader foram feitas com a DeepSeek.
