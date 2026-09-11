/** @type {import('next').NextConfig} */
const nextConfig = {
  // The container image copies only what the standalone server needs.
  output: "standalone",
};

export default nextConfig;
