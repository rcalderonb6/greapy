window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  options: {
    renderActions: {
      addArithmatex: [
        155, // arbitrary priority
        (doc) => {
          for (const node of document.querySelectorAll('.arithmatex')) {
            const math = node.textContent;
            const display = node.tagName.toLowerCase() === 'div';
            const script = document.createElement('script');
            script.type = display ? 'math/tex; mode=display' : 'math/tex';
            script.text = math;
            node.replaceWith(script);
          }
        },
        ''
      ]
    }
  }
};
