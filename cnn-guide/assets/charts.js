(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  if (window.mermaid) {
    mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
  }

  var el = document.getElementById('chart-curve');
  if (!el || !window.echarts) return;

  var nEpoch = 40;
  var epochs = [], trainLoss = [], valLoss = [], valLossOverfit = [];

  function noise(e, seed) {
    return Math.sin((e + seed) * 2.7) * 0.022 + Math.sin((e + seed) * 0.9) * 0.014;
  }

  for (var e = 0; e < nEpoch; e++) {
    epochs.push(e);
    var base = 2.08 * Math.exp(-e / 5.5) + 0.26;
    trainLoss.push(+(base * (1 + 0.03) + noise(e, 1)).toFixed(3));
    valLoss.push(+(base * (1 + 0.10) + noise(e, 7)).toFixed(3));

    var vo;
    if (e <= 14) {
      vo = base * (1 + 0.12) + noise(e, 11);
    } else {
      var k = (e - 14) / 26;
      vo = (2.08 * Math.exp(-14 / 5.5) + 0.26) * (1 + 0.12) + k * k * 0.85 + noise(e, 11);
    }
    valLossOverfit.push(+vo.toFixed(3));
  }

  var chart = echarts.init(el, null, { renderer: 'svg' });
  chart.setOption({
    animation: false,
    grid: { left: 56, right: 30, top: 46, bottom: 44 },
    legend: {
      bottom: 0,
      itemWidth: 16, itemHeight: 9,
      textStyle: { color: muted, fontSize: 12.5 },
      data: ['train loss（健康）', 'val loss（健康）', 'val loss（过拟合）']
    },
    tooltip: {
      appendToBody: true,
      trigger: 'axis',
      formatter: function (ps) {
        var s = 'Epoch ' + ps[0].axisValue;
        ps.forEach(function (p) { s += '<br/>' + p.marker + p.seriesName + ': ' + p.value; });
        return s;
      }
    },
    xAxis: {
      type: 'category', data: epochs, name: 'Epoch', nameLocation: 'middle', nameGap: 26,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, interval: 4 },
      nameTextStyle: { color: muted }
    },
    yAxis: {
      type: 'value', name: 'Cross-Entropy Loss',
      min: 0, max: 2.3,
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      axisLabel: { color: muted },
      nameTextStyle: { color: muted, align: 'left' }
    },
    series: [
      {
        name: 'train loss（健康）', type: 'line', data: trainLoss,
        lineStyle: { color: accent, width: 2.5 }, itemStyle: { color: accent },
        symbol: 'none', smooth: 0.25
      },
      {
        name: 'val loss（健康）', type: 'line', data: valLoss,
        lineStyle: { color: accent2, width: 2.2 }, itemStyle: { color: accent2 },
        symbol: 'none', smooth: 0.25
      },
      {
        name: 'val loss（过拟合）', type: 'line', data: valLossOverfit,
        lineStyle: { color: muted, width: 2, type: 'dashed' }, itemStyle: { color: muted },
        symbol: 'none', smooth: 0.25,
        markPoint: {
          symbol: 'circle', symbolSize: 8,
          itemStyle: { color: accent2 },
          label: { show: true, formatter: '分叉点\n(早停位置)', position: 'top', color: accent2, fontSize: 11, lineHeight: 14 },
          data: [{ coord: [14, valLossOverfit[14]] }]
        }
      }
    ]
  });
  window.addEventListener('resize', function () { chart.resize(); });
})();
