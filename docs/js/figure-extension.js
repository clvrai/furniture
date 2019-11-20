(function (extension) {
    if (typeof showdown !== 'undefined') {
        // global (browser or nodejs global)
        extension(showdown);
    } else if (typeof define === 'function' && define.amd) {
        // AMD
        define(['showdown'], extension);
    } else if (typeof exports === 'object') {
        // Node, CommonJS-like
        module.exports = extension(require('showdown'));
    } else {
        // showdown was not found so we throw
        throw Error('Could not find showdown library');
    }
}(function (showdown) {

    var fig = '<figure>' + '<img src="%1" alt="%2" title="%4">' + '<figcaption>%3</figcaption>' + '</figure>';
    var imgRegex = /(?:<p>)?<img.*?src="(.+?)".*?alt="(.*?)"(.*?)\/?>(?:<\/p>)?/gi;

    // loading extension into shodown
    showdown.extension('figure', function () {
        return [
            {
                type: 'output',
                filter: function (text, converter, options) {
                    var tag = fig;

                    return text.replace(imgRegex, function (match, url, alt, rest) {
                        return tag.replace('%1', url).replace('%2', alt).replace('%3', alt).replace('%4', alt);
                    });
                }
            }
        ];
    });
}));