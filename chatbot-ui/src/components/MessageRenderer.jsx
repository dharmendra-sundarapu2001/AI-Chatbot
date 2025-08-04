import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import python from 'highlight.js/lib/languages/python';
import xml from 'highlight.js/lib/languages/xml'; // For HTML/XML
import css from 'highlight.js/lib/languages/css';
import json from 'highlight.js/lib/languages/json';
import bash from 'highlight.js/lib/languages/bash';
import latex from 'highlight.js/lib/languages/latex'; // For LaTeX math (though usually handled by math-specific renderers)

import 'highlight.js/styles/atom-one-dark.css';

// --- NEW IMPORTS FOR MATH RENDERING ---
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css'; // Import KaTeX CSS for styling math

// Register languages
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('html', xml);
hljs.registerLanguage('css', css);
hljs.registerLanguage('json', json);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('latex', latex);

const MessageRenderer = ({ content, imageData, videoData, websearchInfo }) => {
    const [localImageUrl, setLocalImageUrl] = useState(null);
    const [localVideoUrl, setLocalVideoUrl] = useState(null);

    useEffect(() => {
        if (imageData) {
            console.log("Processing image data:", imageData.substring(0, 50) + "..."); // Debug log
            // Determine MIME type from base64 string if not explicitly provided, or assume common types
            const mimeType = imageData.startsWith('/9j/') ? 'image/jpeg' : // JPEG
                                 imageData.startsWith('iVBORw0KGgoAAA') ? 'image/png' : // PNG
                                 imageData.startsWith('UklGR') ? 'image/webp' : // WebP
                                 'image/jpeg'; // Default to jpeg if unsure

            try {
                const blob = base64toBlob(imageData, mimeType);
                const url = URL.createObjectURL(blob);
                setLocalImageUrl(url);
                console.log("Image URL created:", url); // Debug log
                return () => URL.revokeObjectURL(url);
            } catch (error) {
                console.error("Error creating image blob:", error);
                setLocalImageUrl(null);
            }
        } else {
            setLocalImageUrl(null);
        }
    }, [imageData]);

    useEffect(() => {
        if (videoData) {
            // Determine MIME type for video (e.g., 'video/mp4')
            const mimeType = 'video/mp4'; // You might need to infer this if not provided

            const blob = base64toBlob(videoData, mimeType);
            const url = URL.createObjectURL(blob);
            setLocalVideoUrl(url);
            return () => URL.revokeObjectURL(url);
        } else {
            setLocalVideoUrl(null);
        }
    }, [videoData]);

    const base64toBlob = (base64, mimeType) => {
        try {
            const byteCharacters = atob(base64);
            const byteArrays = [];
            for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                const slice = byteCharacters.slice(offset, offset + 512);
                const byteNumbers = new Array(slice.length);
                for (let i = 0; i < slice.length; i++) {
                    byteNumbers[i] = slice.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                byteArrays.push(byteArray);
            }
            return new Blob(byteArrays, { type: mimeType });
        } catch (error) {
            console.error("Error converting base64 to blob:", error);
            throw error;
        }
    };

    // Parse web resources from content
    const parseWebResources = (text) => {
        const webResourceRegex = /🌐 Web Resources:\s*(https?:\/\/[^\s,]+(?:,\s*https?:\/\/[^\s,]+)*)/;
        const match = text.match(webResourceRegex);
        
        if (match) {
            const urls = match[1].split(',').map(url => url.trim());
            return {
                hasWebResources: true,
                urls: urls,
                cleanContent: text.replace(webResourceRegex, '').trim()
            };
        }
        
        return {
            hasWebResources: false,
            urls: [],
            cleanContent: text
        };
    };

    // Parse invoice analysis from content
    const parseInvoiceAnalysis = (text) => {
        const invoicePattern = /Document Type:\s*Invoice/i;
        
        if (!invoicePattern.test(text)) {
            return { hasInvoiceAnalysis: false, cleanContent: text };
        }

        // Extract fields
        const extractField = (pattern, text) => {
            const match = text.match(pattern);
            return match ? match[1].trim() : 'Not detected';
        };

        const extractedFields = {
            invoiceNumber: extractField(/Invoice Number:\s*([^\n]+)/i, text),
            invoiceDate: extractField(/Invoice Date:\s*([^\n]+)/i, text),
            vendorName: extractField(/Vendor Name:\s*([^\n]+)/i, text),
            buyerName: extractField(/Buyer Name:\s*([^\n]+)/i, text),
            items: extractField(/Items:\s*(\[.*?\])/s, text),
            totalAmount: extractField(/Total Amount:\s*([^\n]+)/i, text),
            taxes: extractField(/Taxes:\s*([^\n]+)/i, text),
            paymentDueDate: extractField(/Payment Due Date:\s*([^\n]+)/i, text)
        };

        // Extract validation results with improved parsing
        const validationPattern = /Validation Results:(.*?)Score:\s*(\d+)\/(\d+)\s*\((\d+)%\)/s;
        const validationMatch = text.match(validationPattern);
        
        let validationResults = [];
        let score = { current: 0, total: 6, percentage: 0 };

        if (validationMatch) {
            const validationText = validationMatch[1];
            const validationLines = validationText.split('\n').filter(line => line.trim() && line.includes(':'));
            
            validationResults = validationLines.map(line => {
                // Enhanced parsing for different formats
                const passMatch = line.match(/:\s*pass/i) || line.match(/\[pass\]/i);
                const failMatch = line.match(/:\s*fail/i) || line.match(/\[fail\]/i);
                const missingMatch = line.match(/:\s*missing/i) || line.match(/\[missing\]/i);
                
                let status = 'unknown';
                if (passMatch) status = 'pass';
                else if (failMatch) status = 'fail';
                else if (missingMatch) status = 'missing';

                // Extract the test description - handle both formats
                let description = line.split(':')[0].trim().replace(/^-\s*/, '');
                const details = line.includes('—') ? line.split('—')[1].trim() : 
                              line.includes(' — ') ? line.split(' — ')[1].trim() : '';

                return {
                    description,
                    status,
                    details
                };
            }).filter(item => item.description);

            score = {
                current: parseInt(validationMatch[2]),
                total: parseInt(validationMatch[3]),
                percentage: parseInt(validationMatch[4])
            };
        }

        // Clean content by removing the invoice analysis
        const cleanContent = text.replace(/Document Type: Invoice.*?Score:\s*\d+\/\d+\s*\(\d+%\)/s, '').trim();

        return {
            hasInvoiceAnalysis: true,
            extractedFields,
            validationResults,
            score,
            cleanContent
        };
    };

    const extractDomainName = (url) => {
        try {
            const domain = new URL(url).hostname;
            return domain.replace('www.', '');
        } catch {
            return 'Unknown Source';
        }
    };

    const getFaviconUrl = (url) => {
        try {
            const domain = new URL(url).hostname;
            return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
        } catch {
            return null;
        }
    };

    const WebSourcesComponent = ({ urls }) => (
        <div className="web-sources-container mb-6 p-4 rounded-lg border border-blue-500/20 bg-gradient-to-r from-blue-900/10 to-cyan-900/10 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-3">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-blue-400">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                </svg>
                <h4 className="text-blue-400 font-semibold text-sm">Web Sources</h4>
                <span className="text-xs text-blue-300/70 bg-blue-500/20 px-2 py-1 rounded-full">
                    {urls.length} source{urls.length > 1 ? 's' : ''}
                </span>
            </div>
            
            <div className="grid gap-2">
                {urls.map((url, index) => (
                    <div key={index} className="web-source-item group">
                        <a 
                            href={url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="flex items-center gap-3 p-3 rounded-md border border-neutral-600/50 bg-neutral-800/30 hover:bg-neutral-700/40 hover:border-blue-500/30 transition-all duration-200 group-hover:shadow-lg group-hover:shadow-blue-500/10"
                        >
                            <div className="flex-shrink-0">
                                <div className="w-8 h-8 rounded-full bg-neutral-700 flex items-center justify-center overflow-hidden">
                                    <img 
                                        src={getFaviconUrl(url)} 
                                        alt=""
                                        className="w-5 h-5"
                                        onError={(e) => {
                                            e.target.style.display = 'none';
                                            e.target.nextSibling.style.display = 'block';
                                        }}
                                    />
                                    <svg 
                                        width="16" 
                                        height="16" 
                                        viewBox="0 0 24 24" 
                                        fill="none" 
                                        stroke="currentColor" 
                                        strokeWidth="2" 
                                        className="text-blue-400 hidden"
                                    >
                                        <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                                        <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                                    </svg>
                                </div>
                            </div>
                            
                            <div className="flex-1 min-w-0">
                                <div className="text-white text-sm font-medium truncate group-hover:text-blue-300 transition-colors">
                                    {extractDomainName(url)}
                                </div>
                                <div className="text-neutral-400 text-xs truncate">
                                    {url.length > 60 ? `${url.substring(0, 60)}...` : url}
                                </div>
                            </div>
                            
                            <div className="flex-shrink-0">
                                <svg 
                                    width="16" 
                                    height="16" 
                                    viewBox="0 0 24 24" 
                                    fill="none" 
                                    stroke="currentColor" 
                                    strokeWidth="2" 
                                    className="text-neutral-500 group-hover:text-blue-400 transition-colors"
                                >
                                    <path d="M7 17L17 7"/>
                                    <path d="M7 7h10v10"/>
                                </svg>
                            </div>
                        </a>
                    </div>
                ))}
            </div>
        </div>
    );

    const InvoiceAnalysisComponent = ({ extractedFields, validationResults, score }) => {
        const getStatusIcon = (status) => {
            switch (status) {
                case 'pass':
                    return (
                        <div className="flex items-center justify-center w-7 h-7 bg-green-500 rounded-full flex-shrink-0">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                                <polyline points="20 6 9 17 4 12" />
                            </svg>
                        </div>
                    );
                case 'fail':
                    return (
                        <div className="flex items-center justify-center w-7 h-7 bg-red-500 rounded-full flex-shrink-0">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                                <line x1="18" y1="6" x2="6" y2="18" />
                                <line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                        </div>
                    );
                case 'missing':
                    return (
                        <div className="flex items-center justify-center w-7 h-7 bg-yellow-500 rounded-full flex-shrink-0">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                                <line x1="12" y1="9" x2="12" y2="13" />
                                <circle cx="12" cy="17" r="1" />
                            </svg>
                        </div>
                    );
                default:
                    return (
                        <div className="flex items-center justify-center w-7 h-7 bg-gray-500 rounded-full flex-shrink-0">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                                <circle cx="12" cy="12" r="1" />
                            </svg>
                        </div>
                    );
            }
        };

        const getScoreColor = (percentage) => {
            if (percentage >= 90) return 'text-green-400';
            if (percentage >= 70) return 'text-yellow-400';
            return 'text-red-400';
        };

        const getScoreBgColor = (percentage) => {
            if (percentage >= 90) return 'from-green-900/20 to-green-800/20 border-green-500/30';
            if (percentage >= 70) return 'from-yellow-900/20 to-yellow-800/20 border-yellow-500/30';
            return 'from-red-900/20 to-red-800/20 border-red-500/30';
        };

        return (
            <div className="invoice-analysis-container mb-6 p-6 rounded-lg border border-blue-500/20 bg-gradient-to-r from-blue-900/10 to-indigo-900/10 backdrop-blur-sm">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="flex items-center justify-center w-10 h-10 bg-blue-500 rounded-full">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z"/>
                            <polyline points="14,2 14,8 20,8"/>
                            <line x1="16" y1="13" x2="8" y2="13"/>
                            <line x1="16" y1="17" x2="8" y2="17"/>
                        </svg>
                    </div>
                    <div>
                        <h3 className="text-blue-400 font-bold text-lg">Invoice Analysis</h3>
                        <p className="text-blue-300/70 text-sm">Document processed and validated</p>
                    </div>
                </div>

                {/* Extracted Fields */}
                <div className="mb-6">
                    <h4 className="text-white font-semibold text-base mb-4 flex items-center gap-2">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="3"/>
                            <path d="M12 1v6M12 17v6M4.22 4.22l4.24 4.24M15.54 15.54l4.24 4.24M1 12h6M17 12h6M4.22 19.78l4.24-4.24M15.54 8.46l4.24-4.24"/>
                        </svg>
                        Extracted Fields
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(extractedFields).map(([key, value]) => (
                            <div key={key} className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/30">
                                <div className="text-neutral-400 text-xs uppercase tracking-wide mb-1">
                                    {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                                </div>
                                <div className="text-white text-sm font-medium break-words">
                                    {value || 'Not detected'}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Validation Results */}
                <div className="mb-6">
                    <h4 className="text-white font-semibold text-base mb-4 flex items-center gap-2">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M9 12l2 2 4-4"/>
                            <circle cx="12" cy="12" r="9"/>
                        </svg>
                        Validation Results
                    </h4>
                    <div className="space-y-3">
                        {validationResults.map((result, index) => (
                            <div key={index} className="flex items-start gap-4 p-4 bg-neutral-800/40 rounded-lg border border-neutral-600/30 hover:border-neutral-500/50 transition-colors">
                                {getStatusIcon(result.status)}
                                <div className="flex-1 min-w-0">
                                    <div className="text-white text-sm font-medium mb-2 leading-relaxed">
                                        {result.description}
                                    </div>
                                    {result.details && (
                                        <div className="text-neutral-300 text-xs bg-neutral-700/30 px-3 py-2 rounded-md border border-neutral-600/20">
                                            <span className="font-medium text-neutral-200">Details:</span> {result.details}
                                        </div>
                                    )}
                                </div>
                                <div className="flex-shrink-0">
                                    <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                                        result.status === 'pass' ? 'bg-green-500/20 text-green-300 border border-green-500/30' :
                                        result.status === 'fail' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                        result.status === 'missing' ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30' :
                                        'bg-gray-500/20 text-gray-300 border border-gray-500/30'
                                    }`}>
                                        {result.status.toUpperCase()}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Score Display */}
                <div className={`bg-gradient-to-r ${getScoreBgColor(score.percentage)} rounded-lg p-6 border backdrop-blur-sm`}>
                    <div className="flex items-center justify-between">
                        <div>
                            <h4 className="text-white font-bold text-lg mb-1">Overall Score</h4>
                            <div className="flex items-center gap-2">
                                <span className={`text-2xl font-bold ${getScoreColor(score.percentage)}`}>
                                    {score.current}/{score.total}
                                </span>
                                <span className="text-neutral-400 text-sm">
                                    ({score.percentage}%)
                                </span>
                            </div>
                        </div>
                        <div className="flex items-center justify-center">
                            <div className="relative w-16 h-16">
                                {/* Progress Circle */}
                                <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                                    <circle
                                        cx="32"
                                        cy="32"
                                        r="28"
                                        stroke="currentColor"
                                        strokeWidth="4"
                                        fill="transparent"
                                        className="text-neutral-600"
                                    />
                                    <circle
                                        cx="32"
                                        cy="32"
                                        r="28"
                                        stroke="currentColor"
                                        strokeWidth="4"
                                        fill="transparent"
                                        strokeDasharray={`${2 * Math.PI * 28}`}
                                        strokeDashoffset={`${2 * Math.PI * 28 * (1 - score.percentage / 100)}`}
                                        className={getScoreColor(score.percentage)}
                                        strokeLinecap="round"
                                    />
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <span className={`text-lg font-bold ${getScoreColor(score.percentage)}`}>
                                        {score.percentage}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    const CodeBlock = ({ inline, className, children }) => {
        const match = /language-(\w+)/.exec(className || '');
        const lang = match ? match[1] : '';

        // --- ADDED THIS CHECK FOR MATH BLOCKS ---
        // If it's a LaTeX/Math block, don't highlight with hljs; let KaTeX handle it.
        // Also apply specific styling for math blocks.
        if (lang === 'latex' || lang === 'math') {
            return (
                <pre className="math-block bg-neutral-900 p-4 rounded my-2 overflow-x-auto text-lg">
                    {/* Render raw children as KaTeX will interpret it later */}
                    <code>{String(children).trim()}</code>
                </pre>
            );
        }

        if (inline) {
            return <code className="bg-neutral-700/50 rounded px-1 text-sm">{children}</code>;
        }

        const highlightedCode = hljs.getLanguage(lang)
            ? hljs.highlight(String(children).trim(), { language: lang }).value
            : hljs.highlightAuto(String(children).trim()).value;

        return (
            <pre className="bg-neutral-800 p-4 rounded my-2 overflow-x-auto text-sm">
                <code dangerouslySetInnerHTML={{ __html: highlightedCode }} />
            </pre>
        );
    };

    const MarkdownImage = ({ alt, src, title }) => {
        const isExternalImage = /(https?:\/\/\S+\.(jpg|jpeg|png|gif|webp|svg)(\?\S*)?)/i.test(src);
        if (isExternalImage) {
            return (
                <div className="my-4">
                    <h4 className="text-sm text-gray-400 mb-2">Embedded Images:</h4>
                    <img
                        src={src}
                        alt={alt}
                        title={title}
                        className="max-w-full h-auto rounded-md border border-neutral-700"
                        onError={(e) => console.error("Failed to load embedded image:", e)}
                    />
                </div>
            );
        }
        return null;
    };

    const MarkdownVideo = ({ src, title }) => {
        const isExternalVideo = /(https?:\/\/\S+\.(mp4|webm|ogg|mov|avi|flv|wmv)(\?\S*)?)/i.test(src);
        if (isExternalVideo) {
            return (
                <div className="my-4">
                    <h4 className="text-sm text-gray-400 mb-2">Embedded Videos:</h4>
                    <video
                        controls
                        className="max-w-full h-auto rounded-md border border-neutral-700"
                        onError={(e) => console.error("Failed to load embedded video:", e)}
                    >
                        <source src={src} type={`video/${src.split('.').pop().split('?')[0].toLowerCase()}`} />
                        Your browser does not support the video tag.
                    </video>
                </div>
            );
        }
        return null;
    };

    // Process content through both parsers in sequence
    const processedContent = (() => {
        // First, remove reference numbers like [1, 2, 3] or [1] from the content
        let cleanedContent = content.replace(/\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g, '');
        
        // First parse web resources
        const webResult = parseWebResources(cleanedContent);
        
        // Then parse invoice analysis from the cleaned content
        const invoiceResult = parseInvoiceAnalysis(webResult.cleanContent);
        
        return {
            webSources: {
                hasWebResources: webResult.hasWebResources,
                urls: webResult.urls
            },
            invoice: {
                hasInvoiceAnalysis: invoiceResult.hasInvoiceAnalysis,
                extractedFields: invoiceResult.extractedFields || {},
                validationResults: invoiceResult.validationResults || [],
                score: invoiceResult.score || { current: 0, total: 6, percentage: 0 }
            },
            finalContent: invoiceResult.cleanContent
        };
    })();

    return (
        <div className="message-content">
            {/* Render uploaded/generated image (if available) FIRST */}
            {localImageUrl && (
                <div className="mb-4">
                    <img
                        src={localImageUrl}
                        alt="Uploaded image"
                        className="max-w-full h-auto rounded-md border border-neutral-700"
                        onLoad={() => console.log("Image loaded successfully")}
                        onError={(e) => {
                            console.error("Failed to load image:", e);
                            console.error("Image URL:", localImageUrl);
                        }}
                    />
                </div>
            )}

            {/* Render invoice analysis if available */}
            {processedContent.invoice.hasInvoiceAnalysis && (
                <InvoiceAnalysisComponent 
                    extractedFields={processedContent.invoice.extractedFields}
                    validationResults={processedContent.invoice.validationResults}
                    score={processedContent.invoice.score}
                />
            )}

            {/* Render markdown content (text description) */}
            <div className="prose prose-invert max-w-none">
                <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                        code: CodeBlock,
                        img: MarkdownImage,
                        a: ({ node, ...props }) => {
                            const videoUrlRegex = /(https?:\/\/\S+\.(mp4|webm|ogg|mov|avi|flv|wmv)(\?\S*)?)/i;
                            if (videoUrlRegex.test(props.href)) {
                                return <MarkdownVideo src={props.href} title={props.title} />;
                            }
                            return <a {...props} />;
                        },
                        p: ({ node, ...props }) => <p {...props}>{props.children}</p>,
                        hr: ({ node, ...props }) => (
                            <hr key={props.key} className="my-4 border-t border-neutral-600" />
                        ),
                        h1: ({ node, ...props }) => <h1 key={props.key} className="text-2xl font-bold my-4">{props.children}</h1>,
                        h2: ({ node, ...props }) => <h2 key={props.key} className="text-xl font-bold my-3">{props.children}</h2>,
                        h3: ({ node, ...props }) => <h3 key={props.key} className="text-lg font-bold my-2">{props.children}</h3>,
                        ul: ({ node, ...props }) => <ul key={props.key} className="ml-4 my-2 list-disc">{props.children}</ul>,
                        ol: ({ node, ...props }) => <ol key={props.key} className="ml-4 my-2 list-decimal">{props.children}</ol>,
                        li: ({ node, ...props }) => <li key={props.key} className="my-1">{props.children}</li>,
                        table: ({ node, ...props }) => (
                            <div className="overflow-x-auto my-4">
                                <table className="min-w-full border border-neutral-600 rounded-lg" {...props} />
                            </div>
                        ),
                        thead: ({ node, ...props }) => <thead className="bg-neutral-800" {...props} />,
                        th: ({ node, ...props }) => <th className="border border-neutral-600 px-4 py-2 text-left text-white font-semibold" {...props} />,
                        tbody: ({ node, ...props }) => <tbody {...props} />,
                        tr: ({ node, ...props }) => <tr className={node.position.start.line % 2 === 0 ? 'bg-neutral-900' : 'bg-neutral-800'} {...props} />,
                        td: ({ node, ...props }) => <td className="border border-neutral-600 px-4 py-2 text-white" {...props} />,
                    }}
                >
                    {processedContent.finalContent}
                </ReactMarkdown>
            </div>

            {/* Render web sources AFTER the response content */}
            {processedContent.webSources.hasWebResources && (
                <WebSourcesComponent urls={processedContent.webSources.urls} />
            )}
            
            {/* Debug info - remove this after testing */}
            {imageData && !localImageUrl && (
                <div className="mt-2 p-2 bg-red-900/20 border border-red-400/20 rounded text-xs">
                    <strong>Debug:</strong> Image data present but failed to create URL. 
                    Data length: {imageData.length}
                </div>
            )}

            {/* Render generated video (if available) after the markdown content */}
            {localVideoUrl && (
                <div className="mt-4">
                    <video
                        controls
                        className="max-w-full h-auto rounded-md border border-neutral-700"
                        onError={(e) => console.error("Failed to load generated video:", e)}
                    >
                        <source src={localVideoUrl} type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>
            )}
        </div>
    );
};

export default MessageRenderer;