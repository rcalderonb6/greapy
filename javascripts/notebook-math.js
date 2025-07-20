// Custom MathJax configuration for notebooks
document.addEventListener('DOMContentLoaded', function() {
    // Additional MathJax configuration for notebooks
    if (typeof MathJax !== 'undefined') {
        // Re-render math when content is dynamically loaded
        MathJax.typesetPromise && MathJax.typesetPromise();
        
        // Handle notebook cell math rendering
        const observer = new MutationObserver(function(mutations) {
            let shouldRerender = false;
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1 && (
                            node.classList && node.classList.contains('arithmatex') ||
                            node.querySelector && node.querySelector('.arithmatex')
                        )) {
                            shouldRerender = true;
                        }
                    });
                }
            });
            
            if (shouldRerender && MathJax.typesetPromise) {
                MathJax.typesetPromise();
            }
        });
        
        // Observe the document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    // Fix for notebook math rendering
    const mathElements = document.querySelectorAll('.highlight pre code');
    mathElements.forEach(function(element) {
        if (element.textContent.includes('$') || element.textContent.includes('\\(')) {
            element.classList.add('language-latex');
        }
    });
});
